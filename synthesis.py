SYNTHESIS_PROMPT = """
You are an expert research analyst and writer. Based on verified research findings about "{query}", 
write a final comprehensive research report.

The report must include:
1. **Overview** — What is this topic about?
2. **Key Findings** — Most important facts, insights, and discoveries.
3. **Current State** — Where things stand today.
4. **Implications** — Why does this matter? What are the real-world impacts?
5. **Conclusion** — A clear, concise takeaway.

- IMPORTANT: Write your entire response in {language}.

Verified Research:
{verified_summary}

Write the final report in {language}:
"""