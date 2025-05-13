from openai import OpenAI

OPEN_API_KEY='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'

client = OpenAI(api_key=OPEN_API_KEY)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "temperature": 0.9,
        "content": "Hello!"
        }
    ]
)    

print(completion)
print('='*80)
print(completion.choices[0].message.content)

# ChatCompletion(id='chatcmpl-CIQ6STUdqKHZkNCzXwgWc8OB5gmjT', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Hello! How can I assist you today?', refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None))], created=1758506280, model='gpt-4o-mini-2024-07-18', object='chat.completion', service_tier='default', system_fingerprint='fp_560af6e559', usage=CompletionUsage(completion_tokens=9, prompt_tokens=9, total_tokens=18, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
# ================================================================================
# Hello! How can I assist you today?

