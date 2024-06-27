from llm_app.patent_analyze import ques_and_ans

question = """I have trading company and looking for technologies that can help me"""

result = ques_and_ans.invoke({"query": question})
qna_response = result["result"].strip()

print("qna_response >>>", qna_response)