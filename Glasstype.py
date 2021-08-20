import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix
 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
 
# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
@st.cache()
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe):
    glass_type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "building windows float processed".upper()
    elif glass_type == 2:
        return "building windows non float processed".upper()
    elif glass_type == 3:
        return "vehicle windows float processed".upper()
    elif glass_type == 4:
        return "vehicle windows non float processed".upper()
    elif glass_type == 5:
        return "containers".upper()
    elif glass_type == 6:
        return "tableware".upper()
    else:
        return "headlamps".upper()

st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis")
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Full Dataset")
    st.dataframe(glass_df)
st.sidebar.subheader("Scatter Plot")

# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the x-axis values:' label
# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.
# Store the current value of this widget in the 'features_list' variable.
features_list = st.sidebar.multiselect("Select the x-axis values:", 
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
st.set_option('deprecation.showPyplotGlobalUse', False)

for feature in features_list:
    st.subheader(f"Scatter plot between {feature} and GlassType")
    plt.figure(figsize = (12, 6))
    sns.scatterplot(x = feature, y = 'GlassType', data = glass_df)
    st.pyplot()
# Add a subheader in the sidebar with label "Visualisation Selector"
st.sidebar.subheader('Visualisation Selector')

# Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
# and with 6 options passed as a tuple ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot').
# Store the current value of this widget in a variable 'plot_types'.
plot_types = st.sidebar.multiselect("Select the charts or plots:", 
                                    ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))            







if 'Histogram' in plot_types:
  st.subheader('Histogram')
  x1 = st.sidebar.selectbox("Select the values for Histrogram:", 
                                        ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize=(10,5))
  plt.title('Histogram')
  plt.hist(glass_df[x1],bins='sturges',edgecolor='black')
  st.pyplot()



if 'Box Plot' in plot_types:
  st.subheader('Box Plot')
  x1 = st.sidebar.selectbox("Select the values for Box Plot:", 
                                        ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize=(10,5))
  plt.title('Box Plot')
  sns.boxplot(glass_df[x1])
  st.pyplot()  




if 'Count Plot' in plot_types:
  st.subheader('Count Plot')
  plt.figure(figsize=(10,5))
  plt.title('Count Plot')
  sns.countplot(glass_df['GlassType'])
  st.pyplot()  


if 'Pair Plot' in plot_types:
  st.subheader('Pair Plot')
  plt.figure(figsize=(10,5))
  plt.title('Pair Plot')
  sns.pairplot(glass_df)
  st.pyplot()  



if 'Correlation Heatmap' in plot_types:
  st.subheader('Correlation Heatmap')
  plt.figure(figsize=(10,5))
  plt.title('Correlation Heatmap')
  
  ax=sns.heatmap(glass_df.corr(),annot=True)
  bottom,top=ax.get_ylim()
  ax.set_ylim(bottom+0.5,top-2)
  st.pyplot()  





if 'Pie Chart' in plot_types:
    st.subheader("Pie Chart")
    pie_data = glass_df['GlassType'].value_counts()
    plt.figure(figsize = (5, 5))
    plt.pie(pie_data, labels = pie_data.index, autopct = '%1.2f%%', 
            startangle = 30, explode = np.linspace(.06, .16, 6))
    st.pyplot()













st.sidebar.subheader("Select your values:")
ri = st.sidebar.slider("Input Ri", float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider("Input Na", float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider("Input Mg", float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider("Input Al", float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider("Input Si", float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider("Input K", float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider("Input Ca", float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider("Input Ba", float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider("Input Fe", float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))




# Add a subheader in the sidebar with label "Choose Classifier"
st.sidebar.subheader("Choose Classifier")

# Add a selectbox in the sidebar with label 'Classifier'.
# and with 2 options passed as a tuple ('Support Vector Machine', 'Random Forest Classifier').
# Store the current value of this slider in a variable 'classifier'.

classifier = st.sidebar.selectbox("Classifier", 
                                 ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))

if classifier == 'Support Vector Machine':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
    kernel_input = st.sidebar.radio("Kernel", ("linear", "rbf", "poly"))
    gamma_input = st. sidebar.number_input("Gamma", 1, 100, step = 1)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
    

    if st.sidebar.button('Classify'):
      st.subheader('SVM')
      object1=SVC(C=c_value,kernel=kernel_input,gamma=gamma_input)
      object1.fit(X_train,y_train)
      predict=object1.predict(X_test)
      accuracy=object1.score(X_test,y_test)
      glass_type = prediction(object1, ri, na, mg, al, si, k, ca, ba, fe)
      st.write('the type of glass predicted is',glass_type)
      st.write('accuracy',accuracy)
      plot_confusion_matrix(object1,X_test,y_test)
      st.pyplot()


if classifier == 'Random Forest Classifier':
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
    max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step = 1)
    if st.sidebar.button('Classify'):
      st.subheader('Random Forest Classifier')
      object2=RandomForestClassifier(n_estimators = n_estimators_input, max_depth=max_depth_input, n_jobs = -1)
      object2.fit(X_train,y_train)
      predict=object2.predict(X_test)
      accuracy=object2.score(X_test,y_test)
      glass_type = prediction(object2, ri, na, mg, al, si, k, ca, ba, fe)
      st.write('the type of glass predicted is',glass_type)
      st.write('accuracy',accuracy)
      plot_confusion_matrix(object2,X_test,y_test)
      st.pyplot()




if classifier == 'Logistic Regression':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C", 1, 100, step = 1)
    max_iter_input = st.sidebar.number_input("Maximum iterations", 10, 1000, step = 10)     
    if st.sidebar.button('Classify'):
      st.subheader('Logistic Regression')
      object3= LogisticRegression(C = c_value, max_iter = max_iter_input)
      object3.fit(X_train,y_train)
      predict=object3.predict(X_test)
      accuracy=object3.score(X_test,y_test)
      glass_type = prediction(object3, ri, na, mg, al, si, k, ca, ba, fe)
      st.write('the type of glass predicted is',glass_type)
      st.write('accuracy',accuracy)
      plot_confusion_matrix(object3,X_test,y_test)
      st.pyplot()    
