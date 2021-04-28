from transformers import pipeline
question_answering = pipeline('question-answering')
context = """
He is eating an apple. He like physics.
"""

question = "What is he eating"
result = question_answering(question=question, context=context)
print("Answer:", result['answer'])
print("Score:", result['score'])
