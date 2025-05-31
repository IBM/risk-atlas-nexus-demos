DRIFT_COT_TEMPLATE = """
        I want you to play the role of a compliance officer and answer the question
        Return the question, answer and explanation in a json format where question, answer and explanation are keys of the json exactly as shown in the examples.
        you should answer the question followed by an explanation on how that answer was generated. 
{% for example in examples %}
        Question: {{ example.question }}
        Answer: {{ example.answer }}
        Explanation: {{ example.explanation }}
{% endfor %}
        Question: {{ prompt }} Consider a binary text classification problem. which class does the prompt belong? Strictly choose one of the two classes. The options are: (1) {{ domain }} or (2) other
"""

RISKS_GENERATION_COT_TEMPLATE = """
        You are are an expert at AI risk classification. I want you to play the role of a risk compliance officer.
        Identify the Risks based on the usecase, question and the answer. Return a list of Risk categories in a json format.
        You should answer followed by an explanation on how that answer was generated. If answer doesn't fit into any of the above categories, classify it as: Unknown.

        Study the JSON below containing list of risk categories and their descriptions. 
        
        RISKS:
        {{ risks }}

        Instructions:
        1. Identify the potential RISKS associated with the given Usecase. Use RISK `description`, `question` and `answer` to verify if the risk is associated with the Usecase.
        2. If Usecase doesn't fit into any of the above RISKS categories, classify it as Unknown.
        3. Respond with a list (top 5 high risks categories) of attribute 'category' containing the risk labels.

{% if examples is not none %}
EXAMPLES:{% for example in examples %}
        Usecase: {{ example.intent }}
        Question: {{ example.question }}
        Answer: {{ example.answer }}
        Risks: {{ example.risks }}{% endfor %}
===== END OF EXAMPLES ======
{% endif %}
        Usecase: {{ usecase }}
        Question: {{ question }}
        Answer: {{ answer }}
"""
