
# mock_round

Components:
Vision Module:

Uses BLIP-2 (Salesforce/blip2-flan-t5-xl) — a state-of-the-art vision-language model.

Takes an image + prompt and generates a descriptive text output about the image.

Language Model Module:

Uses a local HuggingFace LLM pipeline like google/flan-t5-base for text generation.

This handles user queries and reasoning using the context from the vision model.

Tools:

Search Tool: Uses DuckDuckGo search API to get real-time information from the web.

Calculator Tool: Evaluates math expressions.

These augment the agent’s capabilities beyond just text generation.

Agent Integration (LangChain):

Combines vision output, tools, and the LLM into a single agent using LangChain’s agent framework.

The agent uses a “zero-shot” reasoning method to decide when/how to use tools and generate responses.

