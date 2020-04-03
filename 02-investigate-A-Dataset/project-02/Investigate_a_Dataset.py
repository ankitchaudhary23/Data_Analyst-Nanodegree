#!/usr/bin/env python
# coding: utf-8

# > **Tip**: Welcome to the Investigate a Dataset project! You will find tips in quoted sections like this to help organize your approach to your investigation. Before submitting your project, it will be a good idea to go back through your report and remove these sections to make the presentation of your work as tidy as possible. First things first, you might want to double-click this Markdown cell and change the title so that it reflects your dataset and investigation.
# 
# # Project: Investigate a Dataset (Replace this with something more specific!)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > **Tip**: In this section of the report, provide a brief introduction to the dataset you've selected for analysis. At the end of this section, describe the questions that you plan on exploring over the course of the report. Try to build your report around the analysis of at least one dependent variable and three independent variables. If you're not sure what questions to ask, then make sure you familiarize yourself with the dataset, its variables and the dataset context for ideas of what to explore.
# 
# > If you haven't yet selected and downloaded your data, make sure you do that first before coming back here. In order to work with the data in this workspace, you also need to upload it to the workspace. To do so, click on the jupyter icon in the upper left to be taken back to the workspace directory. There should be an 'Upload' button in the upper right that will let you add your data file(s) to the workspace. You can then click on the .ipynb file name to come back here.

# ### Questions to be asked from this movie data are as follows:
# 1. Which movie have highest and lowest profit?
# 2. Which movie have low and high budget?=
# 3. Which production house has produced most movies?
# 4. What is the Number of movies released in each month?
# 5. What is the total profit by year per production house?
# 6. which year has the highest release of movies?
# 7. Year of release vs Profitability?
# 8. Top 10 movies which earn highest profit.

# In[227]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you document your steps carefully and justify your cleaning decisions.
# 
# ### General Properties

# In[228]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data
df = pd.read_csv('tmdb-movies.csv')
print(df.shape)
df.head(3)


# In[229]:


# checking the data types of features
df.dtypes


# In[230]:


df.describe()


# In[231]:


# Checking null values in each feature.
df.isnull().sum()


# In[232]:


df.duplicated().sum()


# > **Tip**: You should _not_ perform too many operations in each cell. Create cells freely to explore your data. One option that you can take with this project is to do a lot of explorations in an initial notebook. These don't have to be organized, but make sure you use enough comments to understand the purpose of each code cell. Then, after you're done with your analysis, create a duplicate notebook where you will trim the excess and organize your steps so that you have a flowing, cohesive report.
# 
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).
# 
# ### Data Cleaning (Replace this with more specific notes!)

# ### Cleaning:
# * There is no need of id feature as we have given imdb_id.
# * Drop id, homepage, tagline, budget_adj, revenue_adj, overview, vote_average
# * We can remove 'tt' from imdb_id feature to convert its data type from object to int for further analysis.
# * Chnage release_date datatype from object to datetime.
# * Change realeae_year datatype from int64 to datetime.* Remove Duplicated value
# * Budget and Revenue column has 0 min . 25%, 50%. how come this is possible 
# * Why runtime have 0 min()
# 
# ### Tidying:
# * There is pipe (|) in cast, genre, production_companies. Solve this issue.
# 

# In[233]:


# Deleting columns which has no significance in analysing the current data for provindg answers to asked questions.
df = df.drop(['id', 'homepage', 'tagline','overview', 'vote_average', 'budget_adj', 'revenue_adj', 'keywords'], axis = 1, inplace=False)


# In[234]:


print(df.shape)
df.head(1)


# In[235]:


df.columns[df.isnull().any()]


# In[236]:


# Display rows with one or more NaN values in dataframe
df[df.isnull().any(axis=1)]


# In[237]:


# Replacing 0 with NAN so that i can drop al na values
df[['budget', 'revenue']] = df[['budget','revenue']].replace(0, np.nan)


# In[238]:


df.dropna(subset=['budget','revenue'], inplace=True)
df.isnull().sum()


# In[239]:


df[df.isnull().any(axis = 1)]


# In[240]:


# Change release_date datatype from object to datetime.
df.release_date = pd.to_datetime(df.release_date)
df.revenue = df.revenue.astype(int)
df.budget = df.budget.astype(int)
# Bit confused if do i really need to change dataype of release_year


# In[241]:


# Making a profit column for answering question regarding profits.
df['profit'] = df['revenue'] - df['budget']
df.profit = df.profit.astype(int)
df.head(1)


# 

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. Compute statistics and create visualizations with the goal of addressing the research questions that you posed in the Introduction section. It is recommended that you be systematic with your approach. Look at one variable at a time, and then follow it up by looking at relationships between variables.

# ### Research Question 1 Which movie have highest and lowest profit?

# In[242]:


df.iloc[[df.profit.idxmax()]]


# In[243]:


df.loc[[df.profit.idxmin()]]


# ### Research Question 2. Which movie have low and high budget?

# In[244]:


df.iloc[[df.budget.idxmax()]]


# In[245]:


df.iloc[[df.budget.idxmin()]]


# ### Research Question 3. Which production house has produced most movies?
# 

# In[246]:


df.production_companies.value_counts().idxmax(),  df.production_companies.value_counts().max()


# ### Research Question 4. What is the Number of movies released in each year? 

# In[247]:


df.release_year.value_counts()


# ### Research Question 5. What is the total profit by year per production house?

# In[248]:


df.groupby(['production_companies', 'release_year'])[['profit']].sum()


# ### Research Question 6. Which year has the highest release of movies?

# In[249]:


top_10 = df.groupby('release_year').count()[['imdb_id']].sort_values('imdb_id',ascending=False).head(25)


# In[250]:


plt.figure(figsize=(12,6), dpi = 130)
plt.plot(top_10)
plt.xlabel('Release Year of Movies in the data set', fontsize = 12)
plt.ylabel('Total Number of movies relased each year', fontsize = 12)
plt.title('Representing Total Number of movies relased each year.');


# ### Research Question 7 : Last 25 Year of release vs Profitability

# In[251]:


profit_years = df.groupby('release_year')['profit'].sum()
profit_years.head(5)


# In[252]:


plt.figure(figsize=(12,6), dpi = 130)
plt.plot(profit_years)
plt.xlabel('Release Year of Movies in the data set', fontsize = 12)
plt.ylabel('Profits earned by Movies', fontsize = 12)
plt.title('Representing Total Profits earned by all movies Vs Year of their release.');


# ### Research Question 8. Top 10 movies which earn highest profit.

# In[253]:


top10_profit = df[['profit']].sort_values('profit',ascending=False)
top10_profit['original_title'] = df['original_title']
top10_profit.head()


# In[260]:


data = list(map(str,(top10_profit['original_title'])))
x = top10_profit['profit'][:10]
y = top10_profit['original_title'][:10]
x,y


# In[269]:


plt.figure(figsize=(10,5), dpi=130)
sns.pointplot(x = x, y = y)

plt.title("Top 10 Profitable Movies",fontsize = 15)
plt.xlabel("Profit",fontsize = 13)
plt.ylabel("Movie Title");


# In[ ]:





# In[ ]:





# <a id='conclusions'></a>
# ## Conclusions
# 
# > **Tip**: Finally, summarize your findings and the results that have been performed. Make sure that you are clear with regards to the limitations of your exploration. If you haven't done any statistical tests, do not imply any statistical conclusions. And make sure you avoid implying causation from correlation!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# ## Submitting your Project 
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[270]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




