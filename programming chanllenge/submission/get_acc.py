f = open("y_pred.txt", "r")
x = f.readlines()

f2 = f = open("y_pred_pipeline.txt", "r")
y = f2.readlines()

matches = 0

for index in range(len(x)):
    if x[index] == y[index]:
        matches += 1

print(matches)