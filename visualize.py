#File: Visualize.py


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np




def DataVisualization(adult):



    """" Load Pre-Processed Data - Need Help here """


input_file = os.path.join('data', 'adult.csv')
output_dir = 'output'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)   


    
    """"Need some Fix here """       "considered adult as the data name"


    
adult = pd.read_csv("adult.data.txt",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.tail()




# Creating a dictionary that contain the education and it's corresponding education level

edu_level = {}
for x,y in adult[['educational-num','education']].drop_duplicates().itertuples(index=False):
    edu_level[y] = x




    

    
    
" Category of Values of Each Data and Histogram     "
    
    
fig = plt.figure(figsize=(20,15))
cols = 5
rows = ceil(float(adult.shape[1]) / cols)
for i, column in enumerate(adult.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if adult.dtypes[column] == np.object:
        adult[column].value_counts().plot(kind="bar", axes=ax)
    else:
        adult[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2) 
    
    
        
    
printmd('## 2.2 Gender vs Income')

gender = round(pd.crosstab(adult.gender, adult.income).div(pd.crosstab(adult.gender, adult.income).apply(sum,1),0),2)
gender.sort_values(by = '>50K', inplace = True)
ax = gender.plot(kind ='bar', title = 'Proportion distribution across gender levels')
ax.set_xlabel('Gender')
ax.set_ylabel('Proportion of population')

printmd('Bar graph showing the proportion of label y(Income) across the sensitive variable(genders) in figure below. From the graph, at an overall view, there exists a wage gap between females and males. Since we do not have the exactly value of the income, we are limited to only observing that the proportion of males earning more than 50k a year is more than double of their female counterparts.')



gender_workclass = round(pd.crosstab(adult.workclass, [adult.income, adult.gender]).div(pd.crosstab(adult.workclass, [adult.income, adult.gender]).apply(sum,1),0),2)
gender_workclass[[('>50K','Male'), ('>50K','Female')]].plot(kind = 'bar', title = 'Proportion distribution across Sensitive variable for each class', figsize = (10,8), rot = 30)
ax.set_xlabel('Gender level')
ax.set_ylabel('Proportion of population')

printmd('Closer look at the disparity in label y and sensitive variable,  across all the working classes as seen in Fig. 3. ')




"""""""""### Gender across working classes """"

 for i in adult.workclass.unique():
        df = adult[adult.workclass == i]

        hours_per_week = round(pd.crosstab(df.gender, df.income).div(pd.crosstab(df.gender, df.income).apply(sum,1),0),2)
        hours_per_week.sort_values(by = '>50K', inplace = True)
        ax = hours_per_week.plot(kind ='bar', title = 'Proportion distribution across Gender for '+ i)
        ax.set_xlabel('Gender')
        ax.set_ylabel('Proportion of population')

        print()

        
        
        
        
printmd('## 2.3. Occupation vs Income')

occupation = round(pd.crosstab(adult.occupation, adult.income).div(pd.crosstab(adult.occupation, adult.income).apply(sum,1),0),2)
occupation.sort_values(by = '>50K', inplace = True)
ax = occupation.plot(kind ='bar', title = 'Proportion distribution across Occupation levels', figsize = (10,8))
ax.set_xlabel('Occupation level')
ax.set_ylabel('Proportion of population')

print()       
        
plt.rcParams['figure.figsize'] = (5, 4)
plt.xlim([20,110])
plt.ylim([20,110])


plt.show()
    