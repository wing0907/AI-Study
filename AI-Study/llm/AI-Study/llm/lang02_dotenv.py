from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경 변수 로드 / 환경변수 키 호출

client = OpenAI()

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

# ChatCompletion(id='chatcmpl-CIQDOrcJHsDznhdM9ekEG1MoBRBmh', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Hello! How can I assist you today?', refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None))], created=1758506710, model='gpt-4o-mini-2024-07-18', object='chat.completion', service_tier='default', system_fingerprint='fp_560af6e559', usage=CompletionUsage(completion_tokens=9, prompt_tokens=9, total_tokens=18, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
# ================================================================================
# Hello! How can I assist you today?

