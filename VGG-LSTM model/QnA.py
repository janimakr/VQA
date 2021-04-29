from transformers import pipeline

qna=pipeline('question-answering')
with open('inputfromDescriptionModel.txt','r') as file:
    des=file.read()
ques=input("Ask your question! \n")
Ans=qna(question=ques,context=des)
print("Answer: ", Ans['answer'])
# print("Score:", Ans['score'])
