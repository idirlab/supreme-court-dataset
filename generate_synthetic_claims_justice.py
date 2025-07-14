from llm import prompt_vllm

prompt = """# Instructions:
Carefully generate a truthful factual claim for this data sample using natural-sounding language while avoiding highly technical language. Keep in mind that this data is only available for countries within the OECD, so avoid absolute statements about the entire world.
{task_prompt}
 
## Output Format:
Return the claim in a JSON object with the following format:
```json
{{
    "claim": "..."
}}
```
 
## Examples:
Here are some example sentences:
{example_sentences}
 
# Dataset Information:

"""