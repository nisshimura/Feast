import os

import openai

KEY = "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"
openai.api_key = KEY


def chat(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # 4
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=2000,
    )
    return completion
