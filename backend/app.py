# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import json
import requests # Still needed for Google Fact Check Tools API

# REMOVED: import asyncio
# REMOVED: import nest_asyncio
# REMOVED: nest_asyncio.apply()

app = Flask(__name__)
CORS(app)

# --- API Keys Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GOOGLE_FACT_CHECK_API_KEY = os.getenv('GOOGLE_FACT_CHECK_API_KEY', '')

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set. Please set it for production.")
if not GOOGLE_FACT_CHECK_API_KEY:
    print("WARNING: GOOGLE_FACT_CHECK_API_KEY environment variable not set. Please set it for production.")

genai.configure(api_key=GEMINI_API_KEY)

# --- Helper function to extract a claim using Gemini (NOW SYNCHRONOUS) ---
def extract_claim_with_gemini(text):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Extract the single most prominent factual claim from the following news content.
        Respond with ONLY the extracted claim text, nothing else. If no clear factual claim is present, respond with "No specific claim identified".

        News Content: "{text}"
        """
        # Using synchronous generate_content()
        response = model.generate_content(prompt)
        
        if response and response.candidates and len(response.candidates) > 0 and \
           response.candidates[0].content and response.candidates[0].content.parts and \
           len(response.candidates[0].content.parts) > 0:
            claim = response.candidates[0].content.parts[0].text.strip()
            print(f"DEBUG: Claim extracted by Gemini (sync): '{claim}'")
            return claim if claim != "No specific claim identified" else None
        print(f"DEBUG: Gemini (sync) did not extract a specific claim from text: '{text}' - Response: {response.text}")
        return None
    except Exception as e:
        print(f"Error extracting claim with Gemini (sync): {e}")
        return None

# --- Helper function to query Google Fact Check Tools API (synchronous, no change) ---
def query_fact_check_api(claim):
    if not GOOGLE_FACT_CHECK_API_KEY:
        print("Google Fact Check API key not set. Skipping external fact check.")
        return None

    api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": claim,
        "key": GOOGLE_FACT_CHECK_API_KEY,
        "languageCode": "en"
    }
    
    try:
        print(f"DEBUG: Querying Fact Check API with claim: '{claim}'")
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"DEBUG: Raw Fact Check API response: {json.dumps(data, indent=2)}")
        
        if data and 'claims' in data and len(data['claims']) > 0:
            first_claim = data['claims'][0]
            claim_review = first_claim.get('claimReview', [])
            if claim_review:
                verdict = claim_review[0].get('textualRating')
                url = claim_review[0].get('url')
                publisher = claim_review[0].get('publisher', {}).get('name', 'Unknown')
                print(f"DEBUG: Fact Check Result: Verdict='{verdict}', URL='{url}', Publisher='{publisher}'")
                return {
                    "verdict": verdict,
                    "url": url,
                    "publisher": publisher
                }
        print("DEBUG: No relevant claims found by Fact Check API.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error querying Google Fact Check API: {e}")
        return None
    except Exception as e:
        print(f"Error processing Fact Check API response: {e}")
        return None


@app.route('/analyze-news', methods=['POST'])
def analyze_news_endpoint(): # NOW SYNCHRONOUS (removed 'async')
    data = request.json
    news_text = data.get('content')
    news_url = data.get('url')

    if not news_text:
        return jsonify({"error": "News content is required."}), 400

    ai_result = {}
    fact_check_result = None
    json_string = ""

    try:
        # Step 1: Perform linguistic analysis with Gemini (NOW SYNCHRONOUS CALL)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        gemini_analysis_prompt = f"""
        You are an AI-powered fake news detection system. Analyze the following news content and, if provided, its source URL, for credibility.
        Focus on:
        - **Linguistic Analysis:** Tone (sensational, neutral, objective), use of emotionally charged language, grammar, spelling, stylistic inconsistencies.
        - **Claim Verifiability (internal):** Are claims presented as facts without evidence? Are sources cited (even if not externally verifiable by you)?
        - **Bias:** Is there a clear slant or agenda?
        - **Completeness:** Does the article present a balanced view or omit crucial context?
        - **Source Reliability (if URL provided):** Comment on the potential reliability suggested by the URL's domain structure (e.g., unusual TLDs, suspicious domain names).

        Provide a structured JSON response with the following fields. Ensure the JSON is valid and contains ONLY the JSON object, without any surrounding text or markdown formatting (e.g., no ```json or ```).
        {{
            "credibilityScore": [integer 0-100, where 0 is highly fake, 100 is highly credible],
            "classification": ["real", "fake", "uncertain"],
            "explanation": "A concise explanation of why this score and classification were given, highlighting key linguistic cues, biases, or lack of verifiable claims.",
            "details": {{
                "sourceReliability": [integer 0-100],
                "contentAnalysis": [integer 0-100],
                "factChecking": [integer 0-100],
                "linguisticAnalysis": [integer 0-100]
            }},
            "disclaimer": "This analysis is based on the provided text and URL. For full fact-checking, independent verification from multiple trusted sources is recommended."
        }}

        News Content:
        "{news_text}"

        News URL (Optional):
        "{news_url if news_url else 'Not provided'}"

        Your response MUST be a valid JSON object and nothing else.
        """

        # Calling synchronous generate_content()
        gemini_response = model.generate_content(gemini_analysis_prompt)
        response_text = gemini_response.text if hasattr(gemini_response, 'text') else ""
        print(f"DEBUG: Raw Gemini analysis response_text: {response_text[:500]}...")

        json_start = response_text.find('{')
        json_end = response_text.rfind('}')

        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_string = response_text[json_start : json_end + 1]
        else:
            print(f"WARNING: Gemini response did not contain a recognizable JSON object for content analysis. Raw response: {response_text}")
            ai_result = {
                "credibilityScore": 50,
                "classification": "uncertain",
                "explanation": "AI content analysis failed: Could not parse Gemini's response (not valid JSON or incomplete).",
                "details": {
                    "sourceReliability": 50, "contentAnalysis": 50, "factChecking": 50, "linguisticAnalysis": 50
                },
                "disclaimer": "AI content analysis failed. For full fact-checking, independent verification from multiple trusted sources is recommended."
            }
            
        try:
            if json_string:
                ai_result = json.loads(json_string)
                if not all(k in ai_result for k in ["credibilityScore", "classification", "explanation", "details"]):
                    print(f"WARNING: AI result missing expected keys after parsing: {ai_result}")
                    ai_result = {
                        "credibilityScore": 50, "classification": "uncertain",
                        "explanation": "AI content analysis returned an unexpected structure (missing keys).",
                        "details": {"sourceReliability": 50, "contentAnalysis": 50, "factChecking": 50, "linguisticAnalysis": 50},
                        "disclaimer": "AI content analysis returned an unexpected structure. For full fact-checking, independent verification from multiple trusted sources is recommended."
                    }
            elif not ai_result:
                 ai_result = {
                    "credibilityScore": 50, "classification": "uncertain",
                    "explanation": "AI content analysis failed: No JSON string found in Gemini's response.",
                    "details": {"sourceReliability": 50, "contentAnalysis": 50, "factChecking": 50, "linguisticAnalysis": 50},
                    "disclaimer": "AI content analysis failed. For full fact-checking, independent verification from multiple trusted sources is recommended."
                }

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from Gemini response: {e}. Raw JSON string: {json_string}")
            ai_result = {
                "credibilityScore": 50, "classification": "uncertain",
                "explanation": "AI content analysis returned malformed JSON.",
                "details": {"sourceReliability": 50, "contentAnalysis": 50, "factChecking": 50, "linguisticAnalysis": 50},
                "disclaimer": "AI content analysis returned malformed JSON. For full fact-checking, independent verification from multiple trusted sources is recommended."
            }

        # Step 2: Extract a claim and perform external fact-check (NOW SYNCHRONOUS CALL)
        extracted_claim = extract_claim_with_gemini(news_text) # Call the synchronous version
        if extracted_claim:
            fact_check_data = query_fact_check_api(extracted_claim)
            if fact_check_data:
                fact_check_result = {
                    "factCheckVerdict": fact_check_data['verdict'],
                    "factCheckUrl": fact_check_data['url'],
                    "factCheckPublisher": fact_check_data['publisher']
                }
                verdict_lower = fact_check_data['verdict'].lower()
                if "false" in verdict_lower or "debunked" in verdict_lower or "misinformation" in verdict_lower:
                    ai_result['classification'] = 'fake'
                    ai_result['credibilityScore'] = min(ai_result['credibilityScore'], 20)
                    ai_result['explanation'] = f"External fact-check confirms this claim is {fact_check_data['verdict']}. " + ai_result['explanation']
                    ai_result['classificationDisplay'] = f"FACT-CHECKED: {fact_check_data['verdict'].upper()}"
                elif "true" in verdict_lower or "verified" in verdict_lower or "accurate" in verdict_lower:
                    ai_result['classification'] = 'real'
                    ai_result['credibilityScore'] = max(ai_result['credibilityScore'], 80)
                    ai_result['explanation'] = f"External fact-check confirms this claim is {fact_check_data['verdict']}. " + ai_result['explanation']
                    ai_result['classificationDisplay'] = f"FACT-CHECKED: {fact_check_data['verdict'].upper()}"
                else:
                    ai_result['classification'] = 'uncertain'
                    ai_result['explanation'] = f"External fact-check verdict: {fact_check_data['verdict']}. " + ai_result['explanation']
                    ai_result['classificationDisplay'] = f"FACT-CHECKED: {fact_check_data['verdict'].upper()}"
            else:
                print("DEBUG: Fact check data not found for extracted claim.")
        else:
            print("DEBUG: No claim extracted, skipping external fact-check.")


        final_result = {**ai_result, **(fact_check_result if fact_check_result else {})}
        print(f"DEBUG: Final result sent to frontend: {json.dumps(final_result, indent=2)}")
        
        return jsonify(final_result), 200

    except Exception as e:
        print(f"Error during AI analysis: {e}")
        return jsonify({"error": f"An internal server error occurred during AI analysis: {str(e)}"}), 500

if __name__ == '__main__':
    # Use Flask's built-in run method. All Gemini calls are now synchronous.
    app.run(host='0.0.0.0', port=5000, debug=True)
