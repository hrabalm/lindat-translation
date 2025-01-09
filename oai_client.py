from openai import OpenAI
server="tdll-3gpu3:8000"
text="this is a test"
client = OpenAI(
    base_url=f"http://{server}/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
    model="CohereForAI/aya-expanse-8b",
    messages=[
        {"role": "user", "content": "Translate the following text into Czech: " + text}
    ]
)
print(completion.choices[0].message.content)
