"""
This project is an AB test for providing an informed decision if implement a new menu in RoundRoaster stores.
The experiment would be run in a selected group of restaurants in Chicago and Denver and this AB test would use matched pairs approach.

Steps to follow are:
1 - Selecting the control and target variables
2 - Match controls for the target stores selected
3 - Compare the results obtained against the same time previous year
4 - Provide a recommendation based on the results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.neighbors import KDTree
from scipy.stats import ttest_rel

def assign_neighbors(df, target_list=[]):
    df.index = df.StoreID
    df.drop('StoreID', axis=True, inplace=True)
    tree = KDTree(df[['weekly_invoices_per_week'
                    ,'avg_sales_per_invoice'
                    ,'weekly_avg_items_per_invoice'
                    , 'AvgMonthSales'
                    , 'Sq_Ft'
                    ,'trend'
                    ,'seasonal']])
    for id in df.index:
        dist, ind = tree.query(df[['weekly_invoices_per_week'
                                ,'avg_sales_per_invoice'
                                ,'weekly_avg_items_per_invoice'
                                , 'AvgMonthSales'
                                , 'Sq_Ft'
                                ,'trend'
                                ,'seasonal']].query(f'index == {id}'), k=10)
        neighbor_n=1
        for i in np.arange(ind.shape[1]):
            index = ind[0][i]
            distance = dist[0][i]
            if neighbor_n == 3:
                break
            else:
                if df.reset_index().at[index,'StoreID'] == id:
                    continue
                elif df.reset_index().at[index,'is_control']:
                    df.at[id,f'neighbor_{neighbor_n}'] = df.reset_index().at[index,'StoreID']
                    df.at[id,f'distance_{neighbor_n}'] = distance
                    neighbor_n +=1
                else:
                    continue
    #Changing data type to integer64 to avoid future problems with the index
    df[['neighbor_1','neighbor_2']] = df[['neighbor_1','neighbor_2']].astype(np.int64)
    return df

def abtest_analysis(control_data, target_data, alpha):
    t_statistic, p_value = ttest_rel(control_data, target_data,alternative='two-sided')
    if p_value < alpha:
        print(f"""For {state} {store} t-test results shows a p-value under your alpha level.\nTherefore, you can reject your Null Hypotesis\np-value:{p_value:.6}\nt-statistic value:{t_statistic:.6}\n""")
    else:
        print(f"""For {state} {store} t-test results shows a p-value under your alpha level.\nTherefore, you CANNOT reject your Null Hypotesis\np-value:{p_value:.6}\nt-statistic value:{t_statistic:.6}\n""")
    return t_statistic, p_value

#Loading the data from the different files provided
df_stores = pd.read_csv('round-roaster-stores (1).csv')
df_stores['is_control'] = True
#Loading targets
df_targets = pd.read_csv('treatment-stores (1).csv')
df_targets['State'] = df_targets['Right_State']
target_units = df_targets['StoreID'].to_list()
#Adding the control logic for classifying the stores
for row_index, row in df_stores.iterrows():
    if row['StoreID'] in target_units:
        df_stores.at[row_index, 'is_control'] = False
    else:
        continue
#Loading transactions data
df_transactions = pd.read_csv('RoundRoastersTransactions.csv')
df_transactions['invoice_date'] = pd.to_datetime(df_transactions['Invoice Date'])
#Filtering the data for having only the data for the dates 
for row_index, row in df_transactions.iterrows():
    if row['invoice_date'].year == 2015:
        week = (np.floor((pd.to_datetime(row['invoice_date'])-pd.to_datetime('2015-04-29')).days / 7 )).astype('int64')
        df_transactions.at[row_index,'week'] =  week
        df_transactions.at[row_index,'week_start'] = pd.to_datetime('2015-04-29') + pd.Timedelta(days=week*7)
    elif row['invoice_date'].year == 2016:
        week = (np.floor((pd.to_datetime(row['invoice_date'])-pd.to_datetime('2016-04-29')).days / 7 )).astype('int64')
        df_transactions.at[row_index,'week'] = week
        df_transactions.at[row_index,'week_start'] = pd.to_datetime('2016-04-29') + pd.Timedelta(days=week*7)
#Dropping useless columns
df_transactions = df_transactions.drop(columns=['Invoice Date','SKU','Category','Product','Size'])
#Calculating the control and target metrics
df_agg = df_transactions.groupby(['StoreID','week', 'week_start','Invoice Number']).agg({'Sales':'sum','QTY':'sum'}).reset_index()\
    .groupby(['StoreID','week', 'week_start']).agg({'Invoice Number':'nunique','Sales':'mean', 'QTY':'mean'}).reset_index()\
    .rename({'Sales':'avg_sales_per_invoice','Invoice Number':'weekly_invoices_per_week','QTY':'weekly_avg_items_per_invoice'},axis=1)
df_agg['year'] = df_agg['week_start'].apply(lambda x: x.year)

#Indexing the dataframe using the datetime field
df_agg.index = df_agg['week_start']
df_agg.drop(columns=['week_start'], inplace=True)
#Calculating trend and seasonality for each store per week
df_results = pd.DataFrame(columns=['StoreID','trend','seasonal'])
for store in df_agg.StoreID.unique():
    results = seasonal_decompose(df_agg.query(f'StoreID == {store}')['weekly_invoices_per_week'], model='additive', extrapolate_trend='freq', period=12)
    df_trend = pd.DataFrame(results.trend, columns=['StoreID','trend'])
    df_trend['StoreID'] = store
    df_seasonal = pd.DataFrame(results.seasonal, columns=['StoreID','seasonal'])
    df_seasonal['StoreID'] = store
    df_results = pd.concat([df_results, df_trend.merge(right=df_seasonal.drop(columns=['StoreID']), left_index=True, right_index=True, how='inner', sort=True)])

#Adding the result columns to the main dataframe
df_agg = pd.merge_ordered(left=df_agg, right=df_results.reset_index(), how='inner',left_on=['StoreID','week_start'], right_on=['StoreID','index']).drop(columns=['index'])
#Then we aggregate seasonal and trend at store level to have the average of each one and, in addition, the average weekly number of invoices too. This last one works as another control variable
df_match = df_agg.query('week_start >= "2015-02-06" & week_start < "2016-07-22"').groupby('StoreID').agg({'weekly_invoices_per_week':'mean'
                                                                             ,'avg_sales_per_invoice':'mean'
                                                                             ,'weekly_avg_items_per_invoice':'mean'
                                                                             ,'trend':'mean'
                                                                             ,'seasonal':'mean'})
#Adding State information to only use stores in the same region
df_stores_il = df_match.merge(df_stores[['StoreID','State','AvgMonthSales','Sq_Ft','is_control']], on='StoreID').query('State == "IL"').drop('State', axis=1)
df_stores_co = df_match.merge(df_stores[['StoreID','State','AvgMonthSales','Sq_Ft','is_control']], on='StoreID').query('State == "CO"').drop('State', axis=1)
#Now we have all our control variables calculated, so we can start looking which 2 control units have a better match with out target units
df_paired_il = assign_neighbors(df_stores_il, target_units)
df_paired_co = assign_neighbors(df_stores_co, target_units)
#Getting the final list of StoreIDs to filter the aggregated dataframe to calculate standard deviation
il_target_stores = df_paired_il.query('is_control == False').index.unique().to_list()
co_target_stores = df_paired_co.query('is_control == False').index.unique().to_list()
#Control Stores 
il_control_stores = df_paired_il.query('is_control == True').index.unique().to_list()
co_control_stores = df_paired_co.query('is_control == True').index.unique().to_list()

#With the paired done, we will run the anaylisis for each stores
df_abtest_results = pd.DataFrame(columns=['is_control'
                                          ,'state'
                                          ,'gross_t_statistic'
                                          ,'gross_p_value'
                                          ,'sales_t_statistic'
                                          ,'sales_p_value']
                                , index=[0,1,2,3])
index = 0
for store_list in [il_control_stores, il_target_stores,co_control_stores,co_target_stores]:
    gross_t_statistic, gross_p_value = abtest_analysis(df_agg.query(f'StoreID in {store_list} & year == 2016 & week >= 0 & week < 12')['weekly_total_gross'].to_list()
                                                        ,df_agg.query(f'StoreID in {store_list} & year == 2015 & week >= 0 & week < 12')['weekly_total_gross'].to_list(),alpha=0.05)
    gross_actual_amount = df_agg.query(f'StoreID in {store_list} & year == 2016 & week >= 0 & week < 12')['weekly_total_gross'].sum()
    gross_baseline = df_agg.query(f'StoreID in {store_list} & year == 2015 & week >= 0 & week < 12')['weekly_total_gross'].sum()
    #Repeating for sales
    sales_t_statistic, sales_p_value = abtest_analysis(df_agg.query(f'StoreID in {store_list} & year == 2016 & week >= 0 & week < 12')['weekly_total_sales'].to_list()
                                                        ,df_agg.query(f'StoreID in {store_list} & year == 2015 & week >= 0 & week < 12')['weekly_total_sales'].to_list(),alpha=0.05)
    sales_actual_amount = df_agg.query(f'StoreID in {store_list} & year == 2016 & week >= 0 & week < 12')['weekly_total_sales'].sum()
    sales_baseline = df_agg.query(f'StoreID in {store_list}& year == 2015 & week >= 0 & week < 12')['weekly_total_sales'].sum()
    if index in [0,1]:
        #Assgining the values to the final dataframe
        df_abtest_results.at[index,'state'] = 'IL'
        if index == 0:
            df_abtest_results.at[index,'is_control'] = True
        else:
            df_abtest_results.at[index,'is_control'] = False
    elif index in [2,3]:
        df_abtest_results.at[index,'state'] = 'CO'
        if index == 0:
            df_abtest_results.at[index,'is_control'] = True
        else:
            df_abtest_results.at[index,'is_control'] = False
    df_abtest_results.at[index,'gross_t_statistic'] = gross_t_statistic
    df_abtest_results.at[index,'gross_p_value'] = gross_p_value
    df_abtest_results.at[index,'sales_t_statistic'] = sales_t_statistic
    df_abtest_results.at[index,'sales_p_value'] = sales_p_value
    #Calculating the lift for gross margin
    gross_lift_total = gross_actual_amount - gross_baseline
    gross_lift_percentage = (gross_lift_total/gross_baseline)*100
    #Calculating the lift for sales
    sales_lift_total = sales_actual_amount - sales_baseline
    sales_lift_percentage = (sales_lift_total/sales_baseline)*100
    #Adding the values to the dataframe
    #Sales
    df_abtest_results.at[index,'gross_lift_total'] = gross_lift_total
    df_abtest_results.at[index,'gross_lift_percentage'] = gross_lift_percentage
    #Sales
    df_abtest_results.at[index,'sales_lift_total'] = sales_lift_total
    df_abtest_results.at[index,'sales_lift_percentage'] = sales_lift_percentage
    #Passing the new index value for the next iteration
    index += 1

df_abtest_results_per_store = pd.DataFrame(columns=['store_id'
                                          ,'state'
                                          ,'is_control'
                                          ,'gross_t_statistic'
                                          ,'gross_p_value'
                                          ,'sales_t_statistic'
                                          ,'sales_p_value'
                                          ,'weekly_total_gross_mean_diff'
                                          ,'weekly_total_sales_mean_diff'
                                          ,'weekly_invoices_per_week_mean_diff'
                                          ,'avg_sales_per_invoice_mean_diff'
                                          ,'weekly_avg_items_per_invoice_mean_diff']
                                , index=np.arange(0, len(il_target_stores)+len(il_control_stores)+len(co_target_stores)+len(co_control_stores)))
index = 0
for store in il_target_stores+il_control_stores+co_target_stores+co_control_stores:
    state = df_stores.query(f'StoreID == {store}')['State'].unique()[0]
    gross_t_statistic, gross_p_value = abtest_analysis(df_agg.query(f'StoreID == {store} & year == 2016 & week >= 0 & week < 12')['weekly_total_gross'].to_list(),df_agg.query(f'StoreID == {store} & year == 2015  & week >= 0 & week < 12')['weekly_total_gross'].to_list(),alpha=0.05)
    sales_t_statistic, sales_p_value = abtest_analysis(df_agg.query(f'StoreID == {store} & year == 2016 & week >= 0 & week < 12')['weekly_total_sales'].to_list(),df_agg.query(f'StoreID == {store} & year == 2015  & week >= 0 & week < 12')['weekly_total_gross'].to_list(),alpha=0.05)
    #Assgining the values to the final dataframe
    df_abtest_results_per_store.at[index,'store_id'] = store
    df_abtest_results_per_store.at[index,'state'] = state
    if store in il_target_stores+co_target_stores:
        df_abtest_results_per_store.at[index,'is_control'] = False
    else:
        df_abtest_results_per_store.at[index,'is_control'] = True
    df_abtest_results_per_store.at[index,'gross_t_statistic'] = gross_t_statistic
    df_abtest_results_per_store.at[index,'gross_p_value'] = gross_p_value
    df_abtest_results_per_store.at[index,'sales_t_statistic'] = sales_t_statistic
    df_abtest_results_per_store.at[index,'sales_p_value'] = sales_p_value
    #Getting the average difference in the main metrics
    for metric in ['weekly_total_gross', 'weekly_total_sales', 'weekly_invoices_per_week','avg_sales_per_invoice','weekly_avg_items_per_invoice']:
        metric_diff = df_agg.query(f'StoreID == {store} & year == 2016 & week >= 0 & week < 12')[metric].mean()-df_agg.query(f'StoreID == {store} & year == 2015 & week >= 0 & week < 12')[metric].mean()
        df_abtest_results_per_store.at[index,f'{metric}_mean_diff'] = metric_diff
    #Passing the new index value for the next iteration
    index += 1

#Storing results
df_abtest_results.to_csv('ab_general_results.csv',sep=',', index=False)
df_abtest_results_per_store.to_csv('ab_general_results_per_store.csv',sep=',', index=False)
df_agg.to_csv('store_transactions_agg.csv',sep=',', index=False)
""""
It's clear that the addition to the menu has a benefit effect on sales and gross margin in all the target stores.
It's expected to have a sales lift of 52.47% in IL stores and a 46.35% in CO
If we talk about the gross margin, in IL we will expect around 42.73% lift and 39.80% in CO.

Considering this information, we recommend to add the new dishes in any store as the gross margin increase is much more higher that the cost in the marketing campaign (18%).
This is clear for IL and CO and, it was mentioned they are good representations of the rest of the store all over the country, we can conclude it's a safe move for the company.
The result, after discounting the marketing budget increase, is to have gross margin lift of 34.47% in IL and 28.35% in CO.
"""
