SUMMARIZATION_PROMPT = """
You are a research summarization expert. Below are raw results fetched from multiple sources about the topic: "{query}".

Your job:
- Read all the content carefully.
- Write a clear, structured summary covering the key facts, findings, and insights.
- Organize by subtopics if needed.
- Be concise but comprehensive.
- IMPORTANT: Write your entire response in {language}.

Raw Source Data:
{raw_data}

Provide a well-structured summary in {language}:
"""