from transformers import pipeline

qna=pipeline('question-answering')
with open('inputfromDescriptionModel.txt','r') as file:
    des=file.read()
print("\n Description generated:",des)
ques=input("Question:")
Ans=qna(question=ques,context=des)
print("Answer: ", Ans['answer'])
# print("Score:", Ans['score'])
