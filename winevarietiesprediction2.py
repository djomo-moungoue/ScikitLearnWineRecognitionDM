from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# 1)
wine = datasets.load_wine()
# print(wine.DESCR)

# 2)
data, labels = wine.data, wine.target

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.7, test_size=0.3, random_state=42)

# print(train_data)
# print(test_data)

# 3)
knn = KNeighborsClassifier()
knn.fit(train_data, train_labels) 

# 4)
predicted = knn.predict(test_data)
print("Predictions from the classifier:")
print(predicted)
print("Target values:")
print(test_labels)

mass = accuracy_score(predicted, test_labels)
print(f"mass: {mass}")

# 5)
class_0,class_1, class_2 = 0, 0, 0

for i in range(0, len(predicted)):
    if predicted[i] == test_labels[i]:
        if predicted[i] == 0:
            class_0 += 1
        elif predicted[i] == 1:
            class_1 += 1
        elif predicted[i] == 2:
            class_2 += 1
                
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
classes = ['class 0', 'class 1', 'class 2']
number_of_correct = [class_0, class_1, class_2]
ax.bar(classes,number_of_correct)
plt.show()
