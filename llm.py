import openai

def prompt_vllm(prompt, model='Qwen/Qwen3-30B-A3B-FP8', think=True):
    client = openai.OpenAI(
        api_key='EMPTY',
        base_url='http://localhost:8000/v1',  # Adjust the base URL if needed
    )

    messages = []
    if not think:
        messages.append({'role': 'system', 'content': '/no_think'})
    messages.append({
        'role': 'user',
        'content': prompt,
    })

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    output = response.choices[0].message.content

    if '<think>' in output[:20]:
        output = output.split('</think>')[-1]
    
    return output
 