from langchain.prompts import PromptTemplate

similarity_template = """
You are Patent Similarity Finder, Given the User Patent Claim and Most Similar Patent, Check how similar they are and tell the user if his idea is unique or not,
if not give him some ideas/scope of improvement

Most Similar Patent: {context}

User Claim: {question}

Conclusion: """

similarity_chain = PromptTemplate(
    input_variables=["context", "question"], template=similarity_template
)

qna_template = """
You are Patent Technology Transfer guide, Given the user's use case and top 2 similar patents, tell him how he can use the patent \
to solve the use case.

Use Case: {question}

Top 2 Similar Patents: {context}

Answer: """

qna_chain = PromptTemplate(
    input_variables=["question", "context"], template=qna_template
)
