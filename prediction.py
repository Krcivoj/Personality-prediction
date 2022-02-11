# -*- coding: utf-8 -*-

#importanje
import pandas as pd
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz

from sklearn.decomposition import PCA #PCA
#import plotly.express as px
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model




#ucitavanje podataka 
data_train = pd.read_csv('Data/train.csv')
data_test = pd.read_csv('Data/test.csv')

data_train.rename(columns={'Personality (Class label)':'Personality'}, inplace=True)
data_test.rename(columns={'Personality (class label)':'Personality'}, inplace=True)
data = pd.concat([data_train, data_test])

flag_valid = 0


#funkcija koja spaja tablice i sređuje podatke
def adjust_data():
  array = data.values
  global flag_valid
  if flag_valid !=0:
      return
  
  flag_valid +=1
  for i in range(len(array)):
    if array[i][0]!='Male' and array[i][0]!='Female':
        data['Gender'].replace(array[i][0],'Female', inplace=True)

  for i in range(len(array)):
    if array[i][1]<15 or array[i][1] > 30:
      data['Age'].replace(array[i][1],round(data['Age'].mean()), inplace=True)

  columns = [ 'Gender','Age','openness','neuroticism','conscientiousness','agreeableness','extraversion', 'Personality']
  for j in range(2,7):
    for i in range(len(array)):
      if array[i][j]<1:
         data[columns[j]].replace(array[i][j],1, inplace=True)
      elif array[i][j] > 8:
        data[columns[j]].replace(array[i][j],8, inplace=True)
                

#ispisuje sve podatke
def all_data_print():
    array = data.values
    for i in range(len(array)):
        print(array[i])
       

#funkcija koja mjenja spolove u brojeve
def gender_to_num():
    adjust_data()
    data['Gender'].replace(['Male'], 0, inplace=True)
    data['Gender'].replace(['Female'], 1, inplace=True)     

#funckija promjene osobnosti u brojeve
def personality_to_num():
    adjust_data()
    data['Personality'].replace(['extraverted'], 1, inplace = True)
    data['Personality'].replace(['serious'], 2, inplace = True)
    data['Personality'].replace(['dependable'], 3, inplace = True)
    data['Personality'].replace(['lively'], 4, inplace = True)
    data['Personality'].replace(['responsible'], 5, inplace = True)

#graf PCA po dvije komponente
def graph_PCA_2():
    gender_to_num()
    features = [ 'Gender','Age','openness','neuroticism','conscientiousness','agreeableness','extraversion']
    n_components = 7

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data[features])

    total_var = pca.explained_variance_ratio_.sum() * 100

    labels = {str(i): f"PC {i+1}" for i in range(n_components)}
    labels['color'] = 'Personality'

    fig = px.scatter_matrix(components, color=data['Personality'], dimensions=range(n_components), labels=labels, title=f'Total Explained Variance: {total_var:.2f}%')
    fig.update_traces(diagonal_visible=False)
    fig.show()

def scree_plot():
    gender_to_num()
    features = [ 'Gender','Age','openness','neuroticism','conscientiousness','agreeableness','extraversion']
    n_components = 7

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data[features])
    
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
    print(per_var)

    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label= labels )
    plt.xlabel('Principal Component')
    plt.ylabel('Persentage of Explaind Variance')
    plt.title('Scree Plot')
    plt.show()

def graph_PCA():
    gender_to_num()
    features = [ 'Gender','Age','openness','neuroticism','conscientiousness','agreeableness','extraversion']
    X = data[features]

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig = px.scatter(components, x=0, y=1, color=data['Personality'])


    for i, feature in enumerate(features):
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1]
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )
        fig.update_layout(title_text= 'PCA', xaxis_title='PC1', yaxis_title='PC2', )
    fig.show()

#heatmap
def graph_heatmap():
    gender_to_num()
    sns.heatmap(data.corr(), vmin=-0.1, vmax=0.1, annot=True, cmap='viridis')
    plt.title('Heatmap', fontsize =20)
    plt.show()

#violinplot
def violin_plot(Y):
  fig = sns.violinplot(data=data, y=Y, x="Personality", hue="Personality", box=True).set_title('Violin plot')
  plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.).set_title('Personality:')
  plt.show()


#dijenje podataka 
def split_data():
    global train 
    global test 
    train_length = round(0.7 * len(data))
    test = data[train_length:]
    train = data[:train_length]


#ispis distribucija
def distribution():
    gender_to_num()
    split_data()
    print('Train distribution:\n', train['Personality'].value_counts() / len(train))
    print('\n\nTest distribution:\n', test['Personality'].value_counts() / len(test))

#funkcija koja priprema podatke za model
def prepare_data():
    adjust_data()
    gender_to_num()
    personality_to_num()
    split_data()
    
    features = [ 'Gender','Age','openness','neuroticism','conscientiousness','agreeableness','extraversion']
    global X
    global y
    global X_test
    global y_test
    X, y = train[features].values, train['Personality'].values
    X_test, y_test = test[features].values, test['Personality'].values

#funckija za tocnost
def accuracy(model, v_X, v_y):
  print("Accuracy:",round(model.score(v_X,v_y)*100,2,), "%")
  
#funkcija za k-validaciju
def k_validation(model, df, k):
  features = [ 'Gender','Age','openness','neuroticism','conscientiousness','agreeableness','extraversion']
  acc = []
  sub_df = split(df,range(int64(ceil(len(df)/k)), len(df), int64(ceil(len(df)/k))))

  for i in range(k): 
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for j in range(k):
      if i!=j:
        df_train = pd.concat([df_train, sub_df[j]])
      else: df_test = sub_df[j]
    
    X, y = df_train[features].values, df_train['Personality'].values
    X_test, y_test = df_test[features].values, df_test['Personality'].values

    model.fit(X,y)

    acc.append(model.score(X_test,y_test)*100)
  
  title = str(type(model)).split('.')[-1].split("'")[0]
  label = 'Average accuracy: {:.2f}%'.format(sum(acc) / len(acc))
  fig = sns.boxplot(y = acc)
  fig.set(ylabel='Accuracy [%]', title= title, xlabel=label)
  plt.show()


#funkcija za odabir parametara
def best_hyperparameters(model, param_grid, X_train, y_train):
  gs = GridSearchCV(model, param_grid)
  gs.fit(X_train, y_train)
  print(gs.best_estimator_)
  return gs.best_estimator_


#funkcija vezana uz model STABLA ODLUKE
def decision_tree():
    prepare_data()
    tree = DecisionTreeClassifier(criterion="entropy")
    tree_param ={
        'splitter': ['best', 'random'],
        'max_features': [None, 'sqrt', 'log2'],
        'class_weight': [None, 'balanced']
        #'min_samples_split' : np.arange(2,10,1),
        #'min_samples_leaf' : np.arange(1,50,1)
        }
    tree = best_hyperparameters(tree, tree_param, X, y)
    tree = tree.fit(X,y)
    accuracy(tree,X,y)

    accuracy(tree,X_test, y_test)
    
    k_validation(tree,data,30)

    
#funkcija za neuronsku mrezu uz vec najbolje parametre
def neural_network():
    prepare_data()
    clf = MLPClassifier(activation='logistic', solver='lbfgs', max_iter=6000, hidden_layer_sizes=(600,600,600), tol=1e-6 )
    clf.fit(X, y)
    accuracy(clf,X,y)
    accuracy(clf,X_test, y_test)
    k_validation(clf, data,30)
    
#funkcija za random forest uz vec najbolje parametre   
def random_forest():
     prepare_data()
     rfc=RandomForestClassifier(criterion='entropy', n_estimators=150, max_depth=350, min_samples_leaf=70, min_samples_split=40)
     rfc.fit(X,y)    
     accuracy(rfc,X_test, y_test)
    
#funkcija za logisticku regresija
def log_reg():
    prepare_data()
    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', max_iter =1000)
    lr_param ={
        #'C':np.arange(0.01, 1.01, 0.01),
        'solver' : ['newton-cg', 'lbfgs'],
        'tol' : np.arange(1e-6,1e-4,0.000001)
        }   
    mul_lr.fit(X, y)
    mul_lr = best_hyperparameters(mul_lr, lr_param, X, y)
    accuracy(mul_lr, X, y)
    accuracy(mul_lr, X_test, y_test)
    











