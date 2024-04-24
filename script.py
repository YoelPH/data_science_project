import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

data = pd.read_csv(r"C:\Users\Jasmine\OneDrive\Documents\Uni\FS2024\Data_science\dementia_patients_health_data.csv", sep=',')

##### DATA EXPLORATION #####
#print(data.head(10))

#we have integers and floats and objects 
#print(data.info()) # -> we have 1000 entries 
#print(data.describe())

integers = data.select_dtypes(int)
#print(integers.columns) # -> 'Diabetic', 'HeartRate', 'Age', 'Cognitive_Test_Scores', 'Dementia'
floats = data.select_dtypes(float)
#print(floats.columns) # -> 'AlcoholLevel', 'BloodOxygenLevel', 'BodyTemperature', 'Weight', 'MRI_Delay', 'Dosage in mg'
objects = data.select_dtypes(object)
#print(objects.columns) # -> 'Prescription', 'Education_Level', 'Dominant_Hand', 'Gender','Family_History', 'Smoking_Status', 'APOE_ε4', 'Physical_Activity','Depression_Status', 'Medication_History', 'Nutrition_Diet','Sleep_Quality', 'Chronic_Health_Conditions'],

#null values 
nulls = data.isnull().sum()
#print(nulls) # -> 515 values in columns Prescription and Dosage in mg -> makes sense, these people take no medication -> give values a name? 

### Graphs of variables ###
print(data.columns)
        
## Demographic data ##
fig, axs = plt.subplots(4, 1, figsize=(10,12))
# Plot histogram of Age distribution
axs[0].hist(data["Age"], bins=20, color='gray')
axs[0].set_title("Histogram of Age distribution")
axs[0].set_xlabel("Age")
axs[0].set_ylabel("Count")

# Plot histogram of Weight distribution
axs[1].hist(data["Weight"], bins=20, color='gray')
axs[1].set_title("Histogram of Weight distribution")
axs[1].set_xlabel("Weight")
axs[1].set_ylabel("Count")

#plot bar plot of gender
gender_counts = data["Gender"].value_counts()
colours = ['#7f7f7f', "#3f3f3f"]
axs[2].bar(gender_counts.index, gender_counts.values, color=colours)
axs[2].set_title("Bar plot of Gender distribution")
axs[2].set_xlabel("Gender")
axs[2].set_ylabel("Count")

#plot bar plot of dominant hand 
hand_counts = data["Dominant_Hand"].value_counts()
colours = ["#7f7f7f", "#3f3f3f"]
axs[3].bar(hand_counts.index, hand_counts.values, color=colours)
axs[3].set_title("Bar plot of Dominant Hand distribution")
axs[3].set_xlabel("Dominant Hand")
axs[3].set_ylabel("Count")

plt.tight_layout()

age_count = (data["Age"].value_counts().sort_index()) #age between 60 and 90
#print(data["Age"].describe()) # mean at 74.91
weight_count = (data["Weight"].value_counts().sort_index()) #weight between 50 and 90 
#print(data["Weight"].describe()) # mean at 74.32
gender_count = (data["Gender"].value_counts()) #504 female, 496 male 
gender_perc = (gender_count/gender_count.sum())*100 #50.4 female, 49.6 male
hand_count = (data["Dominant_Hand"].value_counts()) #519 left and 481 right 
hand_perc = (hand_count/hand_count.sum())*100 #51.9 left, 48.1 right
## 


## HEALTH DATA ##

fig, axs = plt.subplots(5, 1, figsize=(10,12))

#bar plot of diabetic data 
diabetic = data["Diabetic"].value_counts()
colours = ["#7f7f7f", "#3f3f3f"]
axs[0].bar(diabetic.index, diabetic.values, color=colours)
axs[0].set_title("Bar plot of diabetes distribution")
axs[0].set_xlabel("diabetic")
axs[0].set_ylabel("Count")
diabetes_count = data["Diabetic"].value_counts() #513 yes, 487 no
diabetes_perc = diabetes_count/diabetes_count.sum() * 100 #51.3% with, 48.7% without

# Plot histogram of heart rate distribution
axs[1].hist(data["HeartRate"], bins=20, color='gray')
axs[1].set_title("Histogram of heart rate distribution")
axs[1].set_xlabel("heart rate")
axs[1].set_ylabel("Count")
hr_count = data["HeartRate"].value_counts().sort_index() #between 60 and 100 
#print(data["HeartRate"].describe()) # mean 79.383

#plot histogram of alcohol level 
axs[2].hist(data["AlcoholLevel"], bins=20, color='gray')
axs[2].set_title("Histogram of alcohol level distribution")
axs[2].set_xlabel("alcohol level")
axs[2].set_ylabel("Count")
al_count = data["AlcoholLevel"].value_counts().sort_index() #between 0.00 and 0.1999
#print(data["AlcoholLevel"].describe()) # mean 0.0984

#plot histogram of blood oxygen level
axs[3].hist(data["BloodOxygenLevel"], bins=20, color='gray')
axs[3].set_title("Histogram of blood oxygen level distribution")
axs[3].set_xlabel("blood oxygen level")
axs[3].set_ylabel("Count")
bol_count = data["BloodOxygenLevel"].value_counts().sort_index() #between 90.01 and 99.999
#print(data["BloodOxygenLevel"].describe()) #mean 95.23

#plot histogram of body temperature 
axs[4].hist(data["BodyTemperature"], bins=20, color='gray')
axs[4].set_title("Histogram of body temperature distribution")
axs[4].set_xlabel("body temperature")
axs[4].set_ylabel("Count")
bt_count = data["BodyTemperature"].value_counts().sort_index() #between 36.00 and 37.5
#print(data["BodyTemperature"].describe()) # mean 36.76

plt.tight_layout()

##genetic and prescription data##

data["Prescription"].fillna("No Prescription", inplace = True)
presc_count = data["Prescription"].value_counts().sort_index()
presc_perc = presc_count/presc_count.sum() * 100 
#113 Donepezil = 11.3%
#125 Galantamine = 12.5% 
#128 Memantine = 12.8%
#119 Rivastigmine = 51.5%
#515 no prescription = 11.9% 

dementia_data = data[data["Dementia"] == 1]
dem_pres_count = dementia_data["Prescription"].value_counts()
#print(dementia.sum())
#print(dem_pres_count)

#plotting distribution of dosages of the 4 medications  
prescriptions = data["Prescription"].unique()
# Create subplots
fig, axs = plt.subplots(4, 1, figsize=(12, 12))

# Iterate over each prescription and plot histogram
for i, prescription in enumerate(prescriptions[1:]): #we don't want to show "No Prescription" data
    # Filter data for the current prescription
    filtered_data = data[data["Prescription"] == prescription]
    # Plot histogram
    axs[i].hist(filtered_data["Dosage in mg"], bins=20, color="gray")
    axs[i].set_title(f"Dosage of {prescription} for Prescription")
    axs[i].set_xlabel("Dosage")
    axs[i].set_ylabel("Count")
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(4,1, figsize=(10, 12))
columns = ["Medication_History", "Chronic_Health_Conditions", "Family_History", "APOE_ε4"]
for i, col in enumerate(columns): 
    count = data[col].value_counts()
    perc = count / count.sum() * 100 
    
    axs[i].bar(count.index, count.values, color = colours)
    axs[i].set_title(f"Bar plot of {col}")
    axs[i].set_xlabel(col)
    axs[i].set_ylabel("count")
plt.tight_layout()
plt.show()

#history of medication: 514 = 51.4%
#no history of medication: 486 = 48.6% 

#diabetes: 513 = 51.3%
#heart disease: 155 = 15.5%
#hypertension: 153 = 15.3% 
#none: 179: 17.9%

#694 with allel = 69.4%
#306 without allel = 30.6%

#yes family history 520 = 52% 
#no family history 480 = 48%

## lifestyle choices ##
fig, axs = plt.subplots(4,1, figsize=(10, 12))
columns = ["Physical_Activity", "Smoking_Status", "Nutrition_Diet", "Sleep_Quality"]
for i, col in enumerate(columns): 
    count = data[col].value_counts()
    perc = count / count.sum() * 100 
    
    axs[i].bar(count.index, count.values, color = colours)
    axs[i].set_title(f"Bar plot of {col}")
    axs[i].set_xlabel(col)
    axs[i].set_ylabel("count")
plt.tight_layout()
plt.show()

#mild activity: 351 = 35.1%
#moderate activity: 318 = 31.8% 
#sedentary:331 = 33.1%

#former smoker: 458 = 45.8% 
#never smoked: 452 = 45.2%
#current smoker: 90 = 9%

#mediterranean diet: 338 = 33.8% 
#balanced diet: 332 = 33.2% 
#low-carb diet: 330 = 33%

#good sleep: 446 = 46.6% 
#bad sleep: 534 = 53.4%

##cognitive ##
fix, axs = plt.subplots(4,1, figsize=(10,12))
columns = ["Dementia", "Depression_Status", "Cognitive_Test_Scores", "Education_Level"]
for i, col in enumerate(columns): 
    count = data[col].value_counts()
    perc = count / count.sum() * 100 
    
    axs[i].bar(count.index, count.values, color = colours)
    axs[i].set_title(f"Bar plot of {col}")
    axs[i].set_xlabel(col)
    axs[i].set_ylabel("count")
plt.tight_layout()
plt.show()

dementia_gender = dementia_data["Gender"].value_counts()
print(dementia_gender)
#no dementia: 515 = 51.5% 
#dementia: 485 = 48.5% 

#depression: 245 = 24.5% 
#no depression: 755 = 75.5% 

#primary school: 389 = 39.9% 
#secondary school: 304 = 30.4% 
#no school: 155 = 15.5% 
#dimploma / degree: 152 = 15.2% 
    