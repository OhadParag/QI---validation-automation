##############################################################################################################################
###########################################   Beginning - Installing python packages   ################################################ 
##############################################################################################################################

#*** Note: The 2 lines of code below won't be implemented inside Roni's main script since they will be already written 
%reset -f
import pandas as pd
import numpy as np 

#*** Note: The 2 lines of code below won't be implemented inside Roni's main script since the summary and worksheet--> 
#*** dataframes will be already written in the code

## 'Summ' dataframe contains the 'summary' tab and 'ws' dataframe contains the 'worksheet' tab
summ = pd.read_csv('summary.csv') 
ws = pd.read_csv('worksheet.csv') 

#*** As you can see below, all the code is actually one function - 'automatic_validation' 
##############################################################################################################################
#####   'automatic_validation' function has 2 inputs (ws & summ) and 1 output - table with all the validations results  ###### 
##############################################################################################################################

def automatic_validation(summ, ws):


#### Adding 5 columns to ws data frame: 'NLP_mdi', 'nlp_algo', "gt_algo",'fp'&'miss' - will be used for validations below ####

#*** Note: np.select is used instead of for loop
#*** 'NLP_mdi' column has 4 possible values depending on the combination of 'NLP Positive Match' and 'NLP Final Answer'-->
#*** For example: it gets the value 'P-N' if 'NLP Positive Match' is 'Yes' and 'NLP Final Answer' is 'Negative'
#*** Another example: it gets the value 'N-N' if 'NLP Positive Match' is 'No' and 'NLP Final Answer' is 'Negative'
    conditions_nlp_mdi = [(ws['NLP Positive Match'] == 'Yes') & (ws['NLP Final Answer'] == 'Positive'), 
                          (ws['NLP Positive Match'] == 'Yes') & (ws['NLP Final Answer'] == 'Negative'), 
                          (ws['NLP Positive Match'] == 'No')  & (ws['NLP Final Answer'] == 'Positive'), 
                          (ws['NLP Positive Match'] == 'No')  & (ws['NLP Final Answer'] == 'Negative')]
    values_nlp_mdi = ['P-P','P-N','N-P','N-N']
    ws['nlp_mdi'] = np.select(conditions_nlp_mdi, values_nlp_mdi)

#*** 'nlp_algo' column has 4 possible values depending on the combination of 'NLP Positive Match' and 'Is Algo Positive?' -->
#*** For example: it gets the value 'N-T' if 'NLP Positive Match' is 'No' and 'Is Algo Positive?' is True.
    conditions_nlp_algo = [(ws['NLP Positive Match'] == 'Yes') & (ws['Is Algo Positive?'] == True), 
                           (ws['NLP Positive Match'] == 'Yes') & (ws['Is Algo Positive?'] == False), 
                           (ws['NLP Positive Match'] == 'No')  & (ws['Is Algo Positive?'] == True), 
                           (ws['NLP Positive Match'] == 'No')  & (ws['Is Algo Positive?'] == False)]
    values_nlp_algo = ['P-T','P-F','N-T','N-F']
    ws['nlp_algo'] = np.select(conditions_nlp_algo, values_nlp_algo)
    
#*** 'gt_algo' column has 4 possible values depending on the combination of 'NLP Final Answer' and 'Is Algo Positive?' -->
#*** For example: it gets the value 'N-F' if 'NLP Positive Match' is 'No' and 'Is Algo Positive?' is False.  
    conditions_gt_algo = [(ws['NLP Final Answer'] == 'Positive') & (ws['Is Algo Positive?'] == True), 
                           (ws['NLP Final Answer'] == 'Positive') & (ws['Is Algo Positive?'] == False), 
                           (ws['NLP Final Answer'] == 'Negative')  & (ws['Is Algo Positive?'] == True), 
                           (ws['NLP Final Answer'] == 'Negative')  & (ws['Is Algo Positive?'] == False)]
    values_gt_algo = ['P-T','P-F','N-T','N-F']
    ws['gt_algo'] = np.select(conditions_gt_algo, values_gt_algo)
      
#*** 'fp' column gets the value 'bad_fp' if 'Tag (Initial Tag) is not 'PD' and 'Algo Performance' is FP.       
    conditions_fp = [(ws['Tag (Initial Tag)'] != "PD") & (ws['Algo Performance'] == "FP")]
    values_fp = ['bad_fp']
    ws['fp'] = np.select(conditions_fp, values_fp)

#*** 'miss' column gets the value 'Chronic Miss' if 'Dr1 Comment' and 'Dr2 Comment' are 'chronic'.       
    conditions_miss = [(ws['Dr1 Comment'] == "chronic") & (ws['Dr2 Comment'] == "chronic")]
    values_miss = ['Chronic Miss']
    ws['miss'] = np.select(conditions_miss, values_miss)
    
    
    
    
##############################################################################################################################
######################       Creating a variable for each validation:  15 variables for 15 validations         ############### 
##############################################################################################################################

    
                                            #### Category 1 - NLP performance  ####

##  Variable 1: 'P_remain_P' - ratio of NLP P that remained P after tagging out of total NLP P (thus ranges between 0 to 1)
#*** To avoid getting an error in case there are no no 'P-N' values (no cases of 'NLP Positive Match' is 'Yes' and 'NLP Final Answer'
#*** is 'Negative'), a condition that if no instances of P-N the result is 1, is added.
    if ws['nlp_mdi'].eq('P-N').any() == False:
        P_remain_P = 1
    else:
        P_remain_P = ws['nlp_mdi'].value_counts()["P-P"]/(ws['nlp_mdi'].value_counts()["P-P"]+ws['nlp_mdi'].value_counts()["P-N"])

##  Variable 2: 'N_remain_N' - ratio of NLP N that remained N after tagging out of total NLP N (thus ranges between 0 to 1)
#*** To avoid getting an error in case there are no no 'N-P' values (no cases of 'NLP Positive Match' is 'No' and 'NLP Final Answer'
#*** is 'Positive'), a condition that if no instances of N-P the result is 1, is added.
    if ws['nlp_mdi'].eq('N-P').any() == False:
        N_remain_N = 1
    else:
        N_remain_N=ws['nlp_mdi'].value_counts()["N-N"]/(ws['nlp_mdi'].value_counts()["N-N"]+ws['nlp_mdi'].value_counts()["N-P"])

    
                                            ####  Category 2 - MDIs / Tagging ####

## Variable 3: 'highest_pd' -  highest w.p of PD that MDIs found
# ws_pd is a dataframe with only PD values in 'Tag (Initial Tag)' column
# high_pd is a variable with the highest PD value - first we sort ws_pd with highest classifier at first (descending order)--> 
#*** After, we drop duplicates, based on the (['Tag (Initial Tag)']) and since all values of this column are PDs as noted above 
#*** we are left only with the first row. but we don't want the whole row, we only need the value in the column 'Lowest Classifier',
#*** which is why ['Lowest Classifier'] is added. Now we are left with value in 'Lowest Classifier' column and its position/location
#*** This is why we add the iloc[0] and now we have the value of the highest PD' 
#*** Note:As 'highest_pd' is divided by 100 in the end. It is done only to use it in a for loop later. -->
#*** In other words - make code shorter. You can ignore that. it will be multiplied again by 100. 
    ws_pd = ws[ws['Tag (Initial Tag)'] == "PD"]
    high_pd=ws_pd.sort_values('Lowest Classifier', ascending=False).drop_duplicates(['Tag (Initial Tag)'])['Lowest Classifier'].iloc[0]/100

## Variable 4: 'highest_miss': Highest w.p of approved Miss
#*** Exact same explanation/procedure described above for high_pd is relevant high_miss
    ws_miss = ws[ws['Algo Performance'] == "Miss"]
    high_miss=ws_miss.sort_values('Lowest Classifier', ascending=False).drop_duplicates(['Algo Performance'])['Lowest Classifier'].iloc[0]/100


## Variable 5: 'nlp_det' - ratio of nlp_algo 'P-T' out of 'P-T' + 'P-F' (ranges between 0 to 1)
#*** In other words, the ratio of 'Is Algo Positive?' True (out of True+False) when 'NLP Positive Match' is 'Yes' 
    nlp_det = ws['nlp_algo'].value_counts()["P-T"]/(ws['nlp_algo'].value_counts()["P-T"]+ws['nlp_algo'].value_counts()["P-F"])

## Variable 6:  'gt_det' - ratio of gt_algo 'P-T' out of 'P-T' + 'P-F' (ranges between 0 to 1)
#*** In other words, the ratio of 'Is Algo Positive?' True (out of True + False) when 'NLP Final Answer' is 'Yes 
    gt_det = ws['gt_algo'].value_counts()["P-T"]/(ws['gt_algo'].value_counts()["P-T"]+ws['gt_algo'].value_counts()["P-F"])


         
#*** Note about position -  there is a difference of 2 in row values and of 1 in column values between the CSV and code --> 
#*** For example: Total to Review - summ.iloc[42][1] - the cell in row 42 and column 1 when in fact it is in -->
#*** row 44 and column 2 (or B) in the CSV.
#*** The difference of 2 in rows (44 VS. 42) is because in python the headline is not counted + we start counting from 0. 
#*** The difference of 1 in columns (2 VS. 1) is because in python we start counting from 0. 
#*** Also note that all numbers in the summary are actually identified as string in Python, hence the use of 'float' command 
#*** to make them numeric 
            

## Variable 7: rev_delta: delta between 'Total to Review' (summ.iloc[42][1]) to 'Total Reviewed' (summ.iloc[43][1])
    rev_delta = (float(summ.iloc[43][1])-float(summ.iloc[44][1])) / float(summ.iloc[43][1])

    

                                            #### Category 3 - Excluded studies  ####

## Variable 8: exc_rate: Percentage of excluded studies summ.iloc[56][1]) of total included studies (summ.iloc[57][1]). 
    exc_rate= float(summ.iloc[55][1]) / float(summ.iloc[56][1])

## Variable 9: noRep_rate: Percentage of cases with no reports (summ.iloc[49][1]) of total included studies (summ.iloc[57][1]). 
    noRep_rate= float(summ.iloc[50][1])/ float(summ.iloc[56][1])


                                         #### Category 4 - Algo Performance  ####
## Variables 10-13: values in this study of 10) sens, 11) spec, 12) ppv 13) prev
    TP = ws['Algo Performance'].value_counts()["TP"]
    FP = ws['Algo Performance'].value_counts()["FP"]
    TN = ws['Algo Performance'].value_counts()["TN"]
    FN = ws['Algo Performance'].value_counts()["FN"]

    sens= TP / (TP + FN)
    spec= TN / (TN + FP)
    ppv = TP / (TP + FP)
    prev= (TP + FN)/(TP + FN +TN +FP) 

## ci_sens, ci_spec, ci_ppv, ci_prev are the confidence intervals (CI) of sens, spec, ppv and prev in accordance
    ci_sens=1.96*(sens*(1-sens))/float(summ.iloc[1][3])**0.5
    ci_spec=1.96*(spec*(1-spec))/float(summ.iloc[2][3])**0.5
    ci_ppv= 1.96*(ppv *(1-ppv)) /float(summ.iloc[3][1])**0.5
    ci_prev=1.96*(prev*(1-prev))/float(summ.iloc[3][3])**0.5

## Creating 4 variables of values in production of sens, spec, ppv, and prev
#*** For this, we use the globals() built-in function which enables accessing dictionary of global variables which uses -->
#*** strings for keys. Meaning you can create variables with names given as strings at run-time, in this example: 'sens_prod', 
#*** 'spec_prod', 'ppv_prod' and 'prev_prod'. The i is a counter which references to the position in 'summ' data frame --> 
#*** for instance 'sens_prod' variable (which is sensitivity by production) will have, after the for loop, the value in summ.iloc[1][9] and 'spec_prod' variable, 
#*** (which is specificity by production) will have the value of summ.iloc[2][9]. 
    prod = ["sens_prod", "spec_prod", "ppv_prod","prev_prod"]
    i = 1
    for name in prod:
        globals()[name] = round(float(summ.iloc[i][8].replace('%', ''))/100,3)
        i += 1
    
                                             #### Category 4 - Misses and FPs  ####

    
## Variable 14: 'bad_fp' - counts the number of bad_fps in 'fp' column that was added above-->
#*** len(ws.fp) - is the length of the fp column which is the length of ws dataframe, or in other words, the amount of cases in study.
#*** ws.fp.value_counts()[0] counts the number of 0 in fp column, or in other words, the amount of times there is no bad_fp
#*** the difference between len(ws.fp) and ws.fp.value_counts()[0] gives the amount of bad FPs.
#*** note: bad_fp is divided by 100 in the end. It is done only to use it in a for loop later. -->
#*** In other words - make code shorter. You can ignore that. it will be multiplied again by 100. 
    bad_fp = (len(ws.fp)-ws.fp.value_counts()[0])/100
 
 ## Variable 15: 'ch_miss' counts the number of chronic miss in 'miss' column that was added aboe-->
#*** Exact same explanation/procedure described above for bad_fp is relevant for ch_miss
    ch_miss = (len(ws.miss)-ws.miss.value_counts()[0])/100



        
##############################################################################################################################
###########################################                Validation dataframe              ################################# 
##############################################################################################################################
  
                                                #### creating 'validation' data frame ####

#*** 'validation' dataframe is a table with 15 validations and their results.
#*** The rows (index) are the names of the validations. Each of this validations has 4 columns --> 
#*** 1. Value. The values will be the 15 variables that were created above.
#*** 2. '95% CI' -  for some of the validations, a 95% CI is written. 
#*** 3. Category - a string. Each validation belongs to the same specific category.
#*** 4. Result - each validation can be  either .a) ‘V’, meaning it is according to expected by production / above normal. This is what we aim for   
#*** b. below expected by production / above expected by production / below normal. Not good.  
    validation = pd.DataFrame(index=['NLP-P that remained P after tagging (%)','NLP-N that remained N after tagging (%)',
                                     'Highest W.P of PD','Highest W.P of Miss','NLP detected by Algo (%)','GT detected by Algo (%)',
                                     'Delta Between "To review" and "Reviewed" (%)', 'Exclusion Rate (%)', 
                                     'No Report Rate (%)', 'Sensitivity (%)','Specificity (%)','PPV (%)','Prevalence (%)', 
                                     "Number of FPs that were not PD", "Number of Misses that are chronic"], 
                              columns=['Value','95% CI','Category','Result'])
    
## 'row' is a list with all the validation names, i.e it is identical to the index naming in 'validation' dataframe above. 
#*** It will be used in the for loop below
    row=['NLP-P that remained P after tagging (%)','NLP-N that remained N after tagging (%)','Highest W.P of PD',
         'Highest W.P of Miss','NLP detected by Algo (%)','GT detected by Algo (%)', 'Delta Between "To review" and "Reviewed" (%)',
         'Exclusion Rate (%)', 'No Report Rate (%)', 'Sensitivity (%)', 'Specificity (%)','PPV (%)','Prevalence (%)', 
         "Number of FPs that were not PD", "Number of Misses that are chronic"]
    

    
                                    ##### 1st column of 'validation' dataframe - values #####           

#*** 'col' is a list with the 15 variables that were created above. It will be used in the for loop below.
    
    col=[P_remain_P, N_remain_N, high_pd, high_miss, nlp_det, gt_det, rev_delta, exc_rate, noRep_rate, sens, spec, ppv, prev, bad_fp, ch_miss]
    
    ## matching between values and names 
    for i in range(0, len(row)):
        validation['Value'][row[i]]=round(col[i],3)*100

#*** After this for loop, each validation (row) now has a value according to its matching variable (col).        


                                     
                                    ##### 2nd column of the validation dataframe - CIs #####  
        
#*** After the for loop below, some validations will have CI -  sens,spec,ppv and prev-->
#*** i.e the 10th-13th rows in 'validation dataframe (hence 9 until 13 in for loop since we start counting from 0 and we will reach until 12 since we count -1 than the end value in the range function)
#*** 'CI' is a list that contains the CIs variables that were calculated above
#*** K is a counter. when K is equal 0, the CI of the sens will be assigned, when K is 1 - the ci of spec will be assigned
#*** Notice the use of f-string - which is used to print information to the user- in this the value of the CI
    ci=[ci_sens,ci_spec,ci_ppv,ci_prev]

    k=0
    for i in range (9,13):
        validation['95% CI'][row[i]]= f"+-{round(ci[k]*100,1)}"
        k+=1

#*** Setting the other validations (those that are not sens, spec, ppv, prev), to get an empty string     
    for i in range(0, len(row)):
        if validation['95% CI'][row[i]]!= validation['95% CI'][row[i]]:
            validation['95% CI'][row[i]]=""
        




                                ##### 3rd column of the validation dataframe - Categories #####  
            
## Assigning for each validation a category.  For example 'Highest W.P of PD' (3rd validation) belongs to a 'MDIs and tagging' category. 
    validation['Category']['NLP-P that remained P after tagging (%)']=     "1) NLP Performance"
    validation['Category']['NLP-N that remained N after tagging (%)']=     "1) NLP Performance"
    validation['Category']['Highest W.P of PD']=                           "2) MDIs and tagging"
    validation['Category']['Highest W.P of Miss']=                         "2) MDIs and tagging"
    validation['Category']['NLP detected by Algo (%)']=                    "2) MDIs and tagging"
    validation['Category']['GT detected by Algo (%)']=                     "2) MDIs and tagging"
    validation['Category']['Delta Between "To review" and "Reviewed" (%)']="2) MDIs and tagging" 
    validation['Category']['Exclusion Rate (%)'] =                         "3) Excluded studies"
    validation['Category']['No Report Rate (%)'] =                         "3) Excluded studies"
    validation['Category']['Sensitivity (%)'] =                            "4) Algo Performance"
    validation['Category']['Specificity (%)'] =                            "4) Algo Performance"
    validation['Category']['PPV (%)'] =                                    "4) Algo Performance"
    validation['Category']['Prevalence (%)'] =                             "4) Algo Performance"
    validation['Category']['Number of FPs that were not PD'] =             "5) Misses and FPs"
    validation['Category']['Number of Misses that are chronic'] =          "5) Misses and FPs"


    
                                ##### 4th column of the validation dataframe - results #####  


## For specific validations in the ‘validation’ data frame -  (\those who have their matching variables in the values column in 'name' list below.
#*** if the value is below the threshold defined in the value list - result will be set to be "Below Normal" Otherwise - 'V'
#*** Notice the use of f-string - which is used to print information to the user- in this case the value of the treshhold 

    name=[P_remain_P,N_remain_N,high_pd,high_miss,nlp_det,gt_det]
    tresh=[0.7,0.95,0.0094,0.0082,0.85,0.85]
    k=0
    for i in range(0, len(tresh)):
        if name[k] < tresh[k]:
            validation['Result'][i]=f"Very low (<{round(tresh[k]*100,3)})"
        else:
            validation ['Result'][i] = "V"
        k+=1            
        
        
## The result of 'Exclusion Rate (%)' validation is "Above Normal (10%)" if exc_rate is above 10%. Otherwise - 'V'
    if exc_rate >=0.1:
        validation ['Result']['Exclusion Rate (%)'] = "Above Normal (10%)"
    else:
          validation ['Result']['Exclusion Rate (%)'] = "V"

## The Result of 'No Report Rate (%)' validation is ‘Above Normal (2%)’, if noRep_rate is above 2%. Otherwise - 'V'
    if noRep_rate>=0.02: 
        validation ['Result']['No Report Rate (%)'] = "Above Normal (2%)"
    else:
        validation ['Result']['No Report Rate (%)'] = "V"



## comparing between observed metrics + CI to expected by production
#*** 'prod' is a list of variables with the values of sens, spec, ppv and prev by production that were calculated above
#*** 'obs' is a list of variables with the values of sens, spec, ppv and prev the were observed in this study and that were calculated above
#*** 'CI' is a list of variables with values of CIs of sens, spec, ppv and prev that was already referenced above -->
#*** (under the headline of Second column of the validation dataframe - CIs)
#*** K is a counter in this for loop, and is set at the beginning to be 0. When it is 0, we compare between the 0th (i.e first) element -->
#*** in the prod list to the 0th element in the obs list + the 0th element of CI list. So actually the equation is if sens_prod is bigger than sens + ci_sens,-->
#*** the validation in the position 9, since i starts at 9 (10th validation - Sensitivity (%))  get the result of -->
#*** 'Below Expected by production' and the value of sensitivity in production is written in parenthesis. If it is lower, the result of the Sensitivity (%) validation is -->
#*** 'Above Expected by production' and the value of sensitivity in production is written in parentheses. Else the result is 'V'. 
#*** After the counter K increases in 1 (from 0 to 1) and i increases in 1 also so it is 10 and so -->
#*** we do the same comparisons for Specificity (%) validation. And after, for PPV (%)	and Prevalence (%).
#*** Notice the use of f-string - which is used to print information to the user- in this the value of the expected metric by production 
    prod=[sens_prod,spec_prod,ppv_prod,prev_prod]
    obs=[sens,spec,ppv,prev]


    k=0
    for i in range(9,13):
        if prod[k]>obs[k]+ci[k]:
            validation['Result'][i]=f"Below Expected by production ({prod[k]*100}%)"
        elif prod[k]<obs[k]-ci[k]:
            validation['Result'][i]=f"Above Expected by production ({prod[k]*100}%)"
        else:
            validation ['Result'][i] = "V"
        k+=1


    
#*** If the delta between "To review" and "Reviewed" (i.e 'rev_delta variable) is above 2%, →
#*** the 'Delta Between "To review" and "Reviewed" (%)' validation will have a result of 'Above Normal (2%)'. Otherwise - 'V'
    if rev_delta>0.02:
        validation ['Result']['Delta Between "To review" and "Reviewed" (%)'] = "Above Normal (2%)"  
    else:
        validation ['Result']['Delta Between "To review" and "Reviewed" (%)'] = "V"

#*** 'Number of FPs that were not PD' validation will get the result "Validate FPs" if the prob_fp variable is higher than 0. Else - "V - No problematic FPs"
    if bad_fp > 0:
        validation ['Result']['Number of FPs that were not PD'] = "Validate FPs"  
    else:
        validation ['Result']['Number of FPs that were not PD'] = "V - No problematic FPs"

#*** 'Number of Misses that are chronic' validation will get the result "Validate Misses" if ch_miss variable is higher than 0. Else - "V - No chronic Misses"
    if ch_miss > 0:
        validation ['Result']['Number of Misses that are chronic'] = "Validate Misses"  
    else:
        validation ['Result']['Number of Misses that are chronic'] = "V - No chronic Misses"        
        
#*** The lines of codes below are made to output the validation dataframe as an excel file. However, since the location of the folder will change for each user, for now they are set to be comments. 
        
    # determining the name of the file
    # file_name = 'Validation.xlsx'
  
    
    
    
    # saving validation as an excel file

    with pd.ExcelWriter('Validation.xlsx') as writer: 

        validation.to_excel(writer, sheet_name='Validation') 
     
    return validation
automatic_validation(summ, ws)











