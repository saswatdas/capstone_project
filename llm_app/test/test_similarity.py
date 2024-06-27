from llm_app.patent_analyze import similarity_search


question = """An apparatus for generating air pressure and high frequency air pulses to a \
garment having an air core located adjacent the body of a person whereby the \
body of the person is subjected to pressure and high frequency pulses."""

result = similarity_search.invoke({"query": question})
similarity_search_response = result["result"].strip()

print("qna_response >>>", similarity_search_response)
