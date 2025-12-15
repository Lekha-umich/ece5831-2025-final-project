# Emotion-Music-Recommendation
Music Recommendation system based on User's facial expression

# Demo:
The demo for this project can be found in youtube link given below:
https://youtu.be/zxJwIxSDY9g


# Project Description:
This project consists of two modules:
- Emotion Recognition
- Music Recommendation

EffecientNetB0 and EffectiennetB3 model are trained for the Emotion Recognition module. FER-13 Dataset which has  over 35000 grayscale images and 7 emotion classes is used for training. 

The music recommendation module uses a preset database of playlists created for different moods. This can be found in /songs directory.

The live video from the camera is passed to the emotion recognition model (EffecientNetB3), this model processes the feed and recognizes user's emotion based on facial expression. The output of this model is passed to the music recommentdation module which fetches the playlist based on the emotion identified.

The output of this system is displayed using a flask based application.

# Features:
- Real time facial expression detection and song recommendations.
- flask based application to view the recommended songs and live emotion recognition

# Running the app:
Flask: 
- Run <code>pip install -r requirements.txt</code> to install all dependencies.
- Run <code>python app.py</code> and give camera permission if asked.
The python version used for this project is python 3.11

# Tech Stack:
- Keras
- Tensorflow
- Flask

# Dataset:
The dataset used for training the models is FER2013 dataset. Models trained on this dataset can classify 7 emotions. The dataset can be found <a href = "https://drive.google.com/drive/folders/1REv-mgJZ4FBJ-aTqJtUssYH7SDsYPkg5?usp=sharing">here</a>.

The dataset is highly imbalanced with happy class having maxiumum representation. This is one factor affecting the accuracy of the results.

# Image Processing and Training:
- The images were normalised, resized to (224,224) and converted of 32.
- Training took around 13 hours locally and resulted in an accuracy of ~71 %

# Current condition:
The entire project works perfectly fine. Live detection gives good results for the facial expression.

# Project Components:
- haarcascade is for face detection.
- camera.py is the module for video streaming, frame capturing, prediction and recommendation which are passed to main.py.
- main.py is the main flask application file.
- index.html in 'templates' directory is the web page for the application. Basic HTML and CSS.
- utils.py is an utility module for video streaming of web camera with threads to enable real time detection.

# Further Work:
- Use vision-transformer or any other deeper neural networks to improve the performance.
- We have tried training EfficientNetB0 by combining FER-13 and affectnet datasets but the results were not good. This was mainly because FER-13 has grayscal images whereas Affectnet has high resolution RGB images this led to model generalizing for dataset-level features instead of facial features in both the datasets. 
- Training on FER-13 alone can yield to lower accuracies because the dataset has huge class imbalance hence it is important that we train the model based on class weights that is giving importance to classes with less samples so that the model can generalize better for them.

# Presentation and Report:
- The presentation for this project can be found in the youtube link given below:
https://youtu.be/JamxVMeprp8

- The report for this project can be found <a href = "https://drive.google.com/file/d/1T136SuQu1H_0DELrkjZ3WLxkdTjr1Cm_/view?usp=sharing">here</a>.

- The slides for this project can be found <a href = "https://docs.google.com/presentation/d/12jRJSHnvtI6_cuqVAbuF7IhIvlrpl15Z/edit?usp=sharing&ouid=117537070864693278233&rtpof=true&sd=true">here</a>.

# About the files:
- The file trails.ipynb has the code for model trained using both FER-13 and Affectnet dataset using EffecientNetB0 model, and "cbam_efficientnetb0_v2_combinedDatasets" is the weight file.
- The file trail3.ipynb is the model trained using EffecientNetB0 model using FER-13 dataset which has good train accurcacy but bad validation accuracy. "cbam_efficientnetb0_v1.h5" is the weight file.
- final-project.ipynb is the final model which we have decided to use for our project, "fer_efficientnet_b3_70pct.kera" is it's weight file.
