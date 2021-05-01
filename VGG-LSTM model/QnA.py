from transformers import pipeline
import sys

qna=pipeline('question-answering')
with open('outputfromDescriptionModel.txt','r') as file:
    des=file.read()
print("\n Description generated:",des)
ques=input("Question:")
Ans=qna(question=ques,context=des)
print("Answer: ", Ans['answer'])
with open('outputfromVQA.txt', 'w') as f:
    sys.stdout = f 
    print(Ans['answer']);
