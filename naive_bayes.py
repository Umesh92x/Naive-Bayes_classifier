import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# grab data set
dataset=pd.read_csv("C://Users//vinod//PycharmProjects//Excercise//Udemy//Social_Network_Ads.csv")
X=dataset.iloc[:,[2,3]]
y=dataset.iloc[:,4]

# Split into test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=0)

# Feature scalling for speed and accuracy
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

# Making Classifier for classification and prediction
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

# predict the value of test
y_pred=classifier.predict(X_test)

# check the correcteness and number of miscorrect
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#plttting the graph,green region for yes,red region for no.
#   &
#
from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train
x1,x2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,
                            stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,
                            stop=X_set[:,1].max()+1,step=0.01))
#plt.plot(x1,x2,marker=".",color='r',linestyle='none')
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),
                                                x2.ravel()]).T)
             .reshape(x1.shape),
             alpha=0.8,cmap=ListedColormap(('red','green')))

#plt.scatter(X_set[:,0],X_set[:,1],cmap=ListedColormap(('red','green')))

for i,j in enumerate([0,1]):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                cmap=ListedColormap(('blue','pink'))(i))


