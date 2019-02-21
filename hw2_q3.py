import pandas as pd
import sklearn.metrics as skm
data=pd.read_csv("./Data/hw2q3_test.csv")
"""
For pessimistic errors
"""
output=[]
for i in range(len(data)):
    if data["Vaso"][i]==True:
        output.append("Yes")
    elif data["MAP"][i]=="High":
        output.append("Yes")
    else:
        output.append("No")
        print(output)
# for i in range(2):
#     for j in range(2):
#         output[i][j]=int(output[i][j])
output=skm.confusion_matrix(data["Sepsis Shock"],output)
print(output)
print("Accuracy ",(output[0][0]+output[1][1])/(output[0][1]+output[1][1]+output[1][0]+output[0][0]))
print("Recall/Sensitivity ",output[0][0]/(output[0][0]+output[1][0]))
print("Precision ",output[0][0]/(output[0][0]+output[0][1]))
print("Specificity ",output[1][0]/(output[1][0]+output[0][1]))
print("F1 Measure", 2*output[0][0]/(2*output[0][0]+output[1][1]+output[0][1]))
"""
For optimistic errors
"""
output=[]
for i in range(len(data)):
    if data["Vaso"][i]==True:
        output.append("Yes")
    elif data["MAP"][i]=="High":
        output.append("Yes")
    elif data["SBP"][i]=="Very High":
        output.append("Yes")
    elif data["SBP"][i]=="High":
        output.append("Yes")
    elif data["SBP"][i] == "Normal":
        output.append("No")
    else: output.append("Yes")
output=skm.confusion_matrix(data["Sepsis Shock"],output)
print(output)
print("Accuracy ",(output[0][0]+output[1][1])/(output[0][1]+output[1][1]+output[1][0]+output[0][0]))
print("Recall/Sensitivity ",output[0][0]/(output[0][0]+output[1][0]))
print("Precision ",output[0][0]/(output[0][0]+output[0][1]))
print("Specificity ",output[1][0]/(output[1][0]+output[0][1]))
print("F1 Measure", 2*output[0][0]/(2*output[0][0]+output[1][1]+output[0][1]))




