anthropic_prompt = """You are tasked with identifying the high-level user features or business functions in a given code summary. This task is crucial for understanding the main functionalities of a software system from a user or business perspective, Understanding the technical specification is a secondary objective.

Here is the code summary you need to analyze:

<code_summary>
{{CODE_SUMMARY}}
</code_summary>

To complete this task, follow these steps:

1. Carefully read through the entire code summary.

2. As you read, look for descriptions of functionalities that relate to what a user can do with the system or what business processes the system supports.

3. Identify patterns or groupings of related functionalities that could be considered a single high-level feature or business function.

4. Ignore low-level technical details, implementation specifics, or internal system processes that don't directly relate to user interactions or business operations.

5. Consider the perspective of an end-user or a business stakeholder when determining what constitutes a high-level feature or function.

6. Create a list of the identified high-level user features or business functions.

When you've completed your analysis, provide your output in the following format:

<identified_features>
1. [First identified feature or function]
2. [Second identified feature or function]
3. [Third identified feature or function]
...
</identified_features>

<explanation>
Provide a brief explanation of your reasoning for identifying these features or functions, and mention any challenges you encountered in the process.
</explanation>

Remember to focus on high-level features that would be meaningful to users or business stakeholders, rather than getting caught up in technical implementation details. Your goal is to provide a clear, concise list of the main functionalities or business processes supported by the system described in the code summary."""