from transformers import pipeline
question_answering = pipeline('question-answering')

with open('inputfromDescriptionModel.txt','r') as file:
    context = file.read()


question = input("Ask your question! \n")
result = question_answering(question=question, context=context)
print("Answer:", result['answer'])
# print("Score:", result['score'])
