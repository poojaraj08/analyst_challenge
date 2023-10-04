#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
awards_my2022 = pd.read_csv('awards_challenge_my2022.csv')
measures_my2022 = pd.read_csv('measure_list_my2022.csv')


# In[2]:


awards_my2022.head()


# In[3]:


measures_my2022.head()


# In[4]:


#make copy of dataframe
awards = awards_my2022.copy()
measure_data = measures_my2022.copy()
#add domain to awards dataframe
awards_data = pd.merge(awards, measure_data, on='measure_code', how='outer')
awards_data.head()


# # Data Cleansing

# In[6]:


#Create Invalid Data Flag
awards_data['Flag'] = pd.Series([])


# In[8]:


#Populate Invalid Data Flag
awards_data.loc[(awards_data['domain'] == 'Clinical') & (awards_data['denominator'] <30),"Flag"]="Invalid"
awards_data.loc[(awards_data['domain'] == 'Patient Experience') & (awards_data['reliability'] <0.70), "Flag"]="Invalid"
awards_data.loc[(awards_data['domain'] == 'Patient Experience') & (awards_data['reliability'].isnull()), "Flag"]="Invalid"
awards_data.loc[(awards_data['rate']).isnull(), "Flag"] ="Invalid"


# In[9]:


#Create df for Invalid Data
invalid_df = awards_data.loc[awards_data['Flag']=="Invalid"]
invalid_df.head()


# # Adjusted Half Scale Rule

# In[10]:


#Filter dataset where invalid flag is not populated
valid_data = awards_data.loc[awards_data['Flag'].isnull()]
valid_data.head()


# 1.	Calculate the Global Measure Average Rate, Minimum, and Maximum values 
# for each individual measure using only VALID rates from 2022. 

# In[11]:


#Calculate global avg, min, and max using valid rates
global_avg_rate = pd.DataFrame(valid_data.groupby(['measure_code'])['rate'].mean())
global_min = pd.DataFrame(valid_data.groupby(['measure_code'])['rate'].min())
global_max = pd.DataFrame(valid_data.groupby(['measure_code'])['rate'].max())


# In[12]:


#Re-Index Data
global_avg_pd = global_avg_rate.reset_index(drop=False)
global_min_pd = global_min.reset_index(drop=False)
global_max_pd = global_max.reset_index(drop=False)

#Merge data to create a summary df to hold global calculated values
summary = pd.merge(global_avg_pd, global_min_pd, on="measure_code")
summary_df = pd.merge(summary, global_max_pd, on="measure_code")

#Rename Columns
summary_df.rename(columns={'rate_y': 'global_min',
                   'rate_x': 'global_avg','rate': 'global_max'},
          inplace=True, errors='raise')
#Merged DF
summary_df.head()


# 2.	For each PO and its measures from 2022, calculate the difference between 
# their valid measure rate and the Global PO Measure Average Rate calculated in the previous step. 

# In[14]:


#Add global average, min, and max to valid_data DF
valid_data_merged = pd.merge(valid_data, summary_df, on='measure_code', how='outer')
#Calculate difference between PO rate and Global Average Rate
valid_data_merged['PO Rate - Global Avg'] = valid_data_merged['rate'] - valid_data_merged['global_avg']


# 3.	Calculate the POʼs Average Measure Rate Difference for each domain’s measure set (clinical quality and patient experience) 
# from the rate differences calculated in the previous step. 

# In[16]:


#Create domain summary dataframe to calculate POʼs Average Measure Rate Difference
PO_Domain = pd.DataFrame(valid_data_merged.groupby(['org_name','domain'])['PO Rate - Global Avg'].mean())
PO_Domain.rename(columns = {'PO Rate - Global Avg':'Average Measure Rate Difference'}, inplace = True)
PO_Domain_Summary = PO_Domain.reset_index(drop=False)


# 4.	Calculate the imputed rate for invalid or missing measures for each PO using the POʼs Average Measure Rate Difference: 
# a. Imputed Rate = Global Measure Average Rate + POʼs Average Measure Rate Difference for the applicable domain
# b. If the Imputed Rate is greater than the Global Measure Rate Maximum, then set the Imputed Rate equal to the Global Measure Rate Maximum. Also, if the Imputed Rate is less than the Global Measure Rate Minimum, then set the Imputed Rate equal to the Global Measure Rate Minimum. 
# 

# In[18]:


#Add PO Average Measure Rate Difference to invalid data df
invalid_df_merged = pd.merge(invalid_df, PO_Domain_Summary, on=['org_name','domain'], how='left')


# In[35]:


#Added Global Average, Global Min, and Global Max to invalid data df
invalid_df_merged2 = pd.merge(invalid_df_merged, summary_df, on=['measure_code'], how='left')


# In[36]:


#Add column for imputed_value; this value will serve as a placeholder that will be compared against
invalid_df_merged2['imputed_value'] = invalid_df_merged2['global_avg'] + invalid_df_merged2['Average Measure Rate Difference']


# In[37]:


#Add imputed_rate column to be populated
invalid_df_merged2['imputed_rate'] = ""
#Create loop to populate imputed_rate column based on criteria
for index, row in invalid_df_merged2.iterrows():
    if row['imputed_value'] > row['global_max']:
        invalid_df_merged2.at[index, 'imputed_rate'] = invalid_df_merged2.at[index, 'global_max']
    elif row['imputed_value'] < row['global_min']:
        invalid_df_merged2.at[index, 'imputed_rate'] = invalid_df_merged2.at[index, 'global_min']
    else:
        invalid_df_merged2.at[index, 'imputed_rate'] = row['global_avg'] + row['Average Measure Rate Difference']
#Convert string to float
invalid_df_merged2['imputed_rate'] = invalid_df_merged2['imputed_rate'].astype(float)


# 5.	If more than half of the PO’s measures in a domain’s measure set is INVALID in 2022, then the PO is not eligible for the Adjusted Half-Scale Rule imputation or for the Excellence in Healthcare Award. Regarding a PO’s eligibility for the Adjusted Half-Scale Rule, if half of the PO’s measures in a domain’s measure set does not equate to a whole number, please round up to the nearest whole number. 

# In[25]:


#Create Inegibility Flag
elig_adj = pd.DataFrame(invalid_df.groupby(['org_name','domain'])['Flag'].count())
elig_adj['total_domain_measures'] = awards_data.groupby(['org_name','domain'])['measure_code'].count()
elig_adj['Ineligible for Adjustment'] = (elig_adj['Flag']/elig_adj['total_domain_measures'])>0.5
elig_for_adj = elig_adj.reset_index(drop=False)
elig_for_adj.head()


# In[39]:


#Add eligibility flag to invalid data df to create a new df called "invalid_flagged"
invalid_df_merged3 = invalid_df_merged2.copy()
invalid_flagged = pd.merge(invalid_df_merged3, elig_for_adj, on=['org_name', 'domain'], how='left')
invalid_flagged.drop(['higher_is_better', 'measure_summary_units', 'Flag_y','total_domain_measures'], axis = 1, inplace = True) 


# In[40]:


#Create empty adjusted_rate column
invalid_flagged['adjusted_rate'] = ""
#Populated adjusted_rate, which applies the eligibility rules and accordingly determines the rate that should be used
for index, row in invalid_flagged.iterrows():
    if row['Ineligible for Adjustment'] == False:
        invalid_flagged.at[index, 'adjusted_rate'] = invalid_flagged.at[index, 'imputed_rate']
    else:
        invalid_flagged.at[index, 'adjusted_rate'] = invalid_flagged.at[index, 'rate']
#Convert object to float
invalid_flagged['adjusted_rate'] = invalid_flagged['adjusted_rate'].astype(float)


# In[41]:


#Imputed Merged Master: merge invalid_flagged data w/ valid_data_merged
#Drop unecessary columns
imputed_master = pd.merge(invalid_flagged, valid_data_merged, on=['org_name','measure_code','domain'], how='outer')
imputed_master.drop(['po_id_x','year_x','reliability_x','Flag_x','measure_summary_units'], axis = 1, inplace = True) 
imputed_master.drop(['numerator_y','denominator_y','rate_y','reliability_y','higher_is_better','global_avg_y', 'global_min_y','global_max_y','PO Rate - Global Avg'], axis = 1, inplace = True)
imputed_master.drop(['po_id_y','year_y'], axis = 1, inplace = True)


# In[43]:


#Populate Adjusted Rate for Valid Data
for index, row in imputed_master.iterrows():
    if math.isnan(row['Flag']):
        imputed_master.at[index, 'adjusted_rate'] = imputed_master.at[index, 'rate_x']


# # After the Adjusted Half-Scale Rule: 

# 1.	Now that all the eligible POs have Imputed Rates for their missing or invalid measure rates in 2022, calculate the POs’ Average Domain Measure Rate across each domain’s measure set (clinical quality and patient experience). The results are akin to composite scores for the POs, and we will call them the PO’s Clinical Quality Achievement Score and Patient Experience Achievement Score. 

# In[45]:


#create mini df to calculate PO Average Domain Measure Rate
cols_selected = ['org_name', 'measure_code','domain','adjusted_rate']
po_df = imputed_master[cols_selected]


# In[48]:


#Calculate Avg domain rate ("Clinical Quality Score" and "Pat Experience Score")
composite = pd.DataFrame(po_df.groupby(['org_name','domain'])['adjusted_rate'].mean())
composite_df = composite.rename(columns={'adjusted_rate': 'po_domain_avg'})


# 2.	Using the POs’ Clinical Quality Achievement Scores and Patient Experience Achievement Scores, find the Median Achievement Scores for the clinical quality and the patient experience domains across all POs with eligible Achievement Scores. 
# 
# 

# In[ ]:


# Clinical Quality Median
clinical = composite_df.loc[(composite_df['domain']=='Clinical')]
cq_median=clinical['po_domain_avg'].median()
#Patient Experience Median
patient = composite_df.loc[(composite_df['domain']=='Patient Experience')]
pat_median=patient['po_domain_avg'].median()


# 3.	Calculate the median for total cost of care. 

# In[ ]:


#Total Cost Score
total_cost = composite_df.loc[(composite_df['domain']=='Cost')]
tc_median=total_cost['po_domain_avg'].median()


# 4.	Winners of the Excellence in Healthcare award have Achievement Scores that are higher than or equal to the Median Achievement Scores for clinical quality and patient experience domains, and less than or equal to the median for the total cost of care domain. 

# In[ ]:


#Create Winner Dataframe
winners_df = composite_df.copy()
winners_df['cq_median'] = cq_median
winners_df['pat_median'] = pat_median
winners_df['tc_median']= tc_median
winners_df['cq_flag'] = ""
winners_df['pat_flag'] = ""
winners_df['tc_flag'] = ""
winners_df['winner'] = ""
winners_df.head()


# In[ ]:


#Create loop to populate flags
for index, row in winners_df.iterrows():
    if row['domain'] == 'Clinical':
        if winners_df.at[index, 'po_domain_avg'] >= winners_df.at[index, 'cq_median']:
            winners_df.at[index, 'cq_flag'] == "Yes"
    elif row['domain'] == 'Patient Experience':
        if winners_df.at[index, 'po_domain_avg'] >= winners_df.at[index, 'pat_median']:
            winners_df.at[index, 'pat_flag'] == "Yes"
    elif row['domain'] == 'Cost':
        if winners_df.at[index, 'po_domain_avg'] <= winners_df.at[index, 'tc_median']:
            winners_df.at[index, 'tc_flag'] == "Yes"
winners_df.head()


# In[ ]:


#Winner Flag Populated
for index, row in winners_df.iterrows():
    if row['cq_flag'] == 'Yes':
        if row['pat_flag'] == 'Yes':
            if row['tc_flag'] == 'Yes':
                winners_df.at[index, 'winner'] == "Yes"

