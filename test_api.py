import requests
import json
import time


def test_api():
    test_question = """
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   success = as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.
"""

    print("=" * 60)
    print("🧪 DATA ANALYST AGENT API TEST")
    print("=" * 60)

    print("📝 Creating test question file...")
    with open("test_question.txt", "w") as f:
        f.write(test_question)
    print(f"✅ Question file created ({len(test_question)} characters)")

    url = "http://localhost:8000/api/"
    print(f"🌐 Target URL: {url}")

    success = True
    with open("test_question.txt", "rb") as f:
        files = {"questions.txt": f}

        try:
            print("\n🚀 Sending request to API...")
            start_time = time.time()
            response = requests.post(url, files=files, timeout=2000000)
            elapsed_time = time.time() - start_time

            print(f"⏱️  Request completed in {elapsed_time:.2f} seconds")
            print(f"📊 Response status: {response.status_code}")
            print(f"📏 Response size: {len(response.content)} bytes")

            if response.status_code == 200:
                try:
                    result = response.json()
                    print("✅ API request successful!")
                    print("\n📤 RESPONSE ANALYSIS:")
                    print(f"   Type: {type(result)}")

                    if isinstance(result, list):
                        print(f"   Array length: {len(result)}")
                        for i, item in enumerate(result):
                            item_type = type(item).__name__
                            if isinstance(item, str) and len(item) > 100:
                                preview = f"{item[:100]}... ({len(item)} chars total)"
                            else:
                                preview = str(item)
                            print(f"   [{i}] {item_type}: {preview}")

                    print(f"\n📋 Full Response:")
                    print(
                        json.dumps(result, indent=2)[:1000] + "..."
                        if len(str(result)) > 1000
                        else json.dumps(result, indent=2)
                    )

                except json.JSONDecodeError:
                    print("❌ Response is not valid JSON")
                    print("Raw response:", response.text[:500])
                    success = False
            else:
                print(f"❌ API request failed with status {response.status_code}")
                print("Response headers:", dict(response.headers))
                print("Response:", response.text)
                success = False

        except requests.exceptions.ConnectionError:
            print(
                "❌ Could not connect to API. Make sure the server is running on localhost:8000"
            )
            print(
                "💡 Try running: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
            )
            success = False
        except requests.exceptions.Timeout:
            print("❌ Request timed out (>200 seconds)")
            success = False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST COMPLETED SUCCESSFULLY!")
    else:
        print("💥 TEST FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    test_api()
