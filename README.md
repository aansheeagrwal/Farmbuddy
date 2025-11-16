<div align="center">
    <h1>AgriSens : Farmbuddy</h1>
</div>


<img width="953" height="422" alt="Major 1" src="https://github.com/user-attachments/assets/98028f8e-00ad-48f9-8661-29eb2cce6b02" />



## Overview

🌾 FarmBuddy – Smart Farming Assistant

FarmBuddy is an AI-powered smart farming system designed to help farmers make informed decisions about crop health, farm monitoring, and environmental analysis.
It combines IoT sensors, machine learning, real-time dashboards, and automation to simplify modern farming and improve productivity.



## Features

- [x] **Smart Crop Recommendation**: Suggests the best crops based on soil nutrients, weather, and past farming patterns using machine learning.
- [x] **Plant Disease Identification**: Utilizes convolutional neural networks (CNNs) to detect and classify plant diseases from uploaded images, enabling timely and accurate intervention.
- [x] **Fertilizer Recommendation**: Delivers data-driven fertilizer prescriptions by evaluating soil parameters and crop nutrient demands, ensuring enhanced growth and higher yields.
- [x] **Today's Weather Forecast**: Supplies real-time meteorological data such as temperature, humidity, and conditions to support informed farm-management decisions.
- [x] **Smart Farming Guide**: Delivers data-driven recommendations for crop planting timelines and management practices, tailored to soil characteristics and weather patterns.
- [x] **User-Friendly Interface**: Provides a streamlined, intuitive UI that allows users to input agricultural parameters and obtain tailored insights on crops, diseases, and fertilizers.


## Datasets

The Farmbuddy project utilizes three essential datasets that enable accurate crop prediction, plant disease detection, and fertilizer recommendations.

- [X] Crop Recommendation Dataset (2,200 rows):
Contains agro-environmental attributes including nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, rainfall, and soil pH. This dataset is used to predict the most suitable crop for given soil and climate conditions.

- [X] Plant Disease Identification Dataset:
Includes 70,295 training images and 17,572 validation images, covering 38 plant diseases across 14 major crops such as Apple, Tomato, Potato, and Grape. These high-quality leaf images are used to train CNN-based plant disease detection models.

- [X] Fertilizer Recommendation Dataset:
Provides detailed soil nutrient values and crop-specific requirements to generate optimized and tailored fertilizer suggestions.

## Dataset Links

- [X] Crop Recommendation Dataset (Kaggle):
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

- [X] Plant Disease Dataset (Kaggle):
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

- [X] Fertilizer Dataset (Kaggle):
https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra

# 📌 Crop Recommendation Model

The Crop Recommendation Model leverages advanced machine learning techniques to identify the most suitable crop for a given set of environmental and soil conditions. By analyzing key parameters such as soil nutrient levels (N, P, K), temperature, humidity, pH, and rainfall, the model delivers personalized crop suggestions that help farmers maximize productivity and ensure sustainable farming.

The system was trained and evaluated using seven classification algorithms, and among them, the Random Forest Classifier achieved the best performance with an impressive accuracy of 99.55%.
This high accuracy ensures that farmers receive reliable, data-driven recommendations, enabling better crop planning, improved yield, and more efficient utilization of resources.

## Dataset

This dataset contains a total of 2200 rows, with 8 columns representing key environmental and soil parameters: Nitrogen, Phosphorous, Potassium, Temperature, Humidity, pH, Rainfall, and Label. The NPK values indicate the nutrient composition of the soil, while temperature, humidity, and rainfall reflect the average environmental conditions. The pH value represents the acidity or alkalinity of the soil. The Label column specifies the crop that is best suited for the given combination of soil nutrients and environmental factors. This Label is the target variable we aim to predict using machine learning.


## Model Architecture
 
For the Crop Recommendation Model, seven classification algorithms were utilized to predict suitable crop recommendations. These algorithms include:

- Decision Tree
- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest (achieved the best accuracy)
- XGBoost
- KNN
  
Each algorithm was trained on a dataset comprising various factors such as soil nutrients, climate conditions, and historical data to provide accurate crop recommendations to farmers.

## Integration

These two models are integrated into a unified Farmbuddy System with Plant Disease Identification. Together, they provide farmers with end-to-end agricultural support: the crop recommendation model suggests the most suitable crops based on soil nutrients, environmental conditions, and weather data, while the disease identification model accurately detects plant diseases through image analysis. By combining these capabilities, the system empowers farmers to make informed decisions, choose the right crops, and manage plant diseases effectively, ultimately enhancing productivity and promoting sustainable farming.


## Results

- Seven different classification algorithms were tested for the crop recommendation task.
- The performance of each algorithm was evaluated based on accuracy.
- Among all algorithms, Random Forest achieved the highest accuracy of 99.55%.
- Random Forest is identified as the most reliable and effective model for predicting suitable crops.
- Table 1 below presents the accuracy of each algorithm, showing the performance comparison:

> [!IMPORTANT]
> The Random Forest algorithm achieved the highest accuracy of 99.55% in crop recommendation, making it the most reliable model for this system.

**Table 1: Accuracy vs Algorithms**

| Algorithm            | Accuracy   |
| --- | :---: |
| Decision Tree        | 90.0       |
| Gaussian Naive Bayes| 99.09      |
| Support Vector Machine (SVM) | 10.68 |
| Logistic Regression  | 95.23      |
| Random Forest        | 99.55      |
| XGBoost              | 99.09      |
| KNN                  | 97.5       |



| Accuracy Comparison Graph of all models |
|---------------------------|
![1](https://github.com/ravikant-diwakar/AgriSens-SMART-CROP-RECOMMENDATION-SYSTEM-WITH-PLANT-DISEASE-IDENTIFICATION/assets/110620635/604bd0b3-5161-48e2-aef0-28267fd85aac)

> The **Accuracy vs Crop Graphs** visualize the performance of different algorithms in crop recommendation accuracy.

| Accuracy vs Crop Graphs | Accuracy vs Crop Graphs |
| ----------- | -------------|
| ![4](https://github.com/user-attachments/assets/ef096a91-ee2f-470e-a134-9c0ba9c4862a) | ![6](https://github.com/user-attachments/assets/84ed33e7-f496-469e-b663-f20997936ced) |
![8](https://github.com/user-attachments/assets/a9230e96-b813-4213-90cc-6654f8cec69f) | ![10](https://github.com/user-attachments/assets/8455aa24-1856-43f9-ab0e-07adeda49dda) |
![12](https://github.com/user-attachments/assets/40d7bdeb-bc4f-40f5-97c3-110e229e30ca) | ![14](https://github.com/user-attachments/assets/639ba618-9930-467d-a462-354c5fd44a9c) |
![3](https://github.com/user-attachments/assets/69a9033f-cf39-45d3-93c9-57fb9ea8229d) | ![2](https://github.com/user-attachments/assets/70ffbe66-ca11-4b89-b2bf-16d1f71b5534) |


---

# 📌Plant Disease Identification Model 

The Plant Disease Identification Model uses Convolutional Neural Networks (CNNs) to accurately detect and classify diseases from plant leaf images. It is trained on the Plant Disease Image Dataset, which contains 70,295 training images and 17,572 validation images, covering 38 disease classes across 14 different crops. The model can identify diseases such as Apple Scab, Tomato Blight, Powdery Mildew, and many more. This enables farmers to detect diseases early and take timely action, helping to protect crops and improve yields.

## Dataset

The Plant Disease Image Dataset, used for crop disease identification, contains 70,295 training images and 17,572 validation images, representing a total of 38 different plant disease classes. All images are standardized to a resolution of 128×128 pixels to ensure uniformity during model training. The complete dataset occupies approximately 5 GB of storage, making it a comprehensive resource for developing accurate and reliable plant disease detection models.


## Model Architecture
   
The Plant Disease Identification Model is built using a Convolutional Neural Network (CNN) architecture specifically designed for image-based disease detection. The CNN processes leaf images through multiple convolutional, pooling, and dense layers to automatically extract features such as texture, color patterns, and disease spots. By leveraging deep learning, the model learns to differentiate between healthy and diseased leaves with high accuracy. This architecture enables precise disease classification, helping farmers identify issues early and take timely action to protect crop health and maximize yield.


### Key Features:
- **Crop Specific**: The model is designed to diagnose diseases for a specific set of crops.
- **Disease Diagnosis**: It can classify diseases based on images of leaves.
- **Accuracy**: The CNN model demonstrates high accuracy in identifying plant diseases, helping farmers and researchers detect issues early.

### Supported Crops and Diseases:
- The model works with a predefined list of 14 crops.
- For each crop, the model is trained to detect and classify up to 38 specific diseases.

> [!NOTE]
> Since model is trained for specific crops only so it can diagnose those specific crops only. The List of Crops For which this model will be helpful is:

```
[ 'Apple',
'Blueberry',
'Cherry_(including sour)',
'Corn_(maize)',
'Grape',
'Orange',
'Peach', 'Pepper, _bell',
'Potato',
'Raspberry',
'Soybean',
'Squash',
'Strawberry',
'Tomato' ]
```

> [!NOTE]
> The crop which can be used for diagnosis can only diagnose specific disease for which the model is trained. The List of crop diseases on Which Model is trained on is:

```
Found 17572 files belonging to 38 classes.
['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

```

### How it Works:
- The model uses images of plant leaves to detect symptoms of various diseases.
- It applies CNN-based image classification to identify the correct disease for a given crop.


> [!IMPORTANT]
> ### System Requirements
>  - Python: Version 3.8 or above
>  - TensorFlow/Keras: For disease identification
>  - Streamlit: For creating the web interface

> [!TIP]
> ### Common Issues and Tips
> - Ensure all dependencies in the `requirements.txt` are installed.
> - For TensorFlow-based disease detection, ensure you have a compatible GPU or CPU for faster processing.




## 📷 Screenshots

| Home page | Features | 
| --------- | --------- |
<img width="953" height="422" alt="Major 1" src="https://github.com/user-attachments/assets/82c0c76b-dca3-4365-803e-a67c52b1789f" />


| Team Members | Contact Us |
| --------- | --------- |
<img width="951" height="422" alt="major8" src="https://github.com/user-attachments/assets/162db94e-810c-4075-a99b-73e997fd61ca" />
<img width="953" height="425" alt="major9" src="https://github.com/user-attachments/assets/b5ee375d-7265-405c-9749-c4b48e5d37c8" />






---

















