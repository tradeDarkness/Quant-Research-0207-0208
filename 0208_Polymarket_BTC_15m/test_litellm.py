
import os
from litellm import completion

# Set env directly to test
os.environ["GEMINI_API_KEY"] = "AIzaSyBvIOFM7YBGkQYEDGgThxZdgbsW2adyhvE"

models_to_test = [
    "gemini/gemini-1.5-flash",
    "gemini/gemini-1.5-pro",
    "gemini/gemini-2.0-flash-exp", 
    "gemini/gemini-3-flash-preview"
]

for model in models_to_test:
    print(f"Testing model: {model}...")
    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": "Hello, are you working?"}],
        )
        print(f"Success with {model}!")
        print(response)
        break
    except Exception as e:
        print(f"Failed with {model}: {e}")
