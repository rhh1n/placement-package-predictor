import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample Dataset
data = {
    'cgpa':[6,7,8,9,7.5,8.5,6.5,9.2,8.8,7.2],
    'projects':[1,2,3,4,2,3,1,5,4,2],
    'internships':[0,1,2,2,1,2,0,3,2,1],
    'coding':[4,6,8,9,7,8,5,9,9,6],
    'communication':[5,6,7,8,6,7,5,8,9,6],
    'package':[3,5,7,12,6,9,4,15,13,5]
}

df = pd.DataFrame(data)

X = df.drop('package',axis=1)
y = df['package']

model = LinearRegression()
model.fit(X,y)

pickle.dump(model,open('model.pkl','wb'))

print("Model Created Successfully")
