#!/usr/bin/env python3

import os
import argparse
import json
import textwrap
import sys

from openai import OpenAI

# Config
TEMPERATURE = 0.0
MAX_TOKENS = 1500

BASE_URL = os.getenv("FAKE_RESPONSE_OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("FAKE_RESPONSE_OPENAI_API_KEY")
MODEL = os.getenv("FAKE_RESPONSE_OPENAI_MODEL", "openai/gpt-4.1-nano")
if OPENAI_API_KEY is None:
    print(
        "ERROR: Set FAKE_RESPONSE_OPENAI_API_KEY environment variable.", file=sys.stderr
    )

client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

PROMPT_INSTRUCTIONS = textwrap.dedent(
    """
You will be given the content of a user's `questions.txt` file describing a data analysis
task and the exact format the user expects the answer to be returned in.

Your job:
1. Produce a FAKE response whose *format* exactly matches what the user requested.
   - The *contents* may be fictional / invented — factual correctness is NOT required.
   - The *format* must be strictly preserved (JSON vs array vs object, field names, order if requested).
2. DO NOT print any explanations, commentary, or extra text. Output only the response content.
3. If the user explicitly requests images/plots encoded as base64 data URIs, include a placeholder
   such as "data:image/png;base64,AAA..." (no need for a real image).
4. If the user requests numeric tolerances or sizes (e.g., "image < 100000 bytes"), you may
   ignore exact size checks — but maintain the requested field name and type.
5. If the question asks for a JSON array or object, your output MUST be valid JSON.
6. If the question asks for a specific structure (e.g., an array of 4 elements, or specific keys),
   follow that structure exactly (use plausible dummy values).
7. If you are uncertain about precise keys, produce a reasonable JSON that captures the intent
   and uses clear placeholder values.

Return only the response (for example: JSON array, JSON object, or plain text exactly as requested).
"""
).strip()


def build_prompt(questions_text: str) -> str:
    return (
        PROMPT_INSTRUCTIONS
        + "\n\n---\n\n<questions.txt>:\n\n"
        + questions_text
        + "</questions.txt>\n\n---\n\nNow produce the FAKE response."
    )


def call_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise assistant that outputs ONLY the requested response.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    text = resp.choices[0].message.content
    return text


def main(questions_path: str):
    if not os.path.exists(questions_path):
        print(f"ERROR: {questions_path} not found.", file=sys.stderr)
        return

    with open(questions_path, "r", encoding="utf-8") as f:
        questions_text = f.read()

    prompt = build_prompt(questions_text)
    print("Sending prompt to LLM (format-only / fake-response request)...")
    lm_output = call_llm(prompt)

    try:
        return json.loads(lm_output)
    except Exception:
        return lm_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ask LLM to produce a fake response that matches required format."
    )
    parser.add_argument(
        "--questions", "-q", default="questions.txt", help="Path to questions.txt"
    )
    args = parser.parse_args()
    print(main(args.questions))
