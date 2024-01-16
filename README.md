# Crop-and-Fertilizer-Recommender
This project helps in identifying the best crop suited for the particular nutrition and climate values along with the best fertilizer to be used for gaining maximum yield.
To do this I have used RandomForestClassfier for both crop and fertilizer recommendation, and for the website I used bootstrap framework for creating a responsive webpage.
To integrate the Machine Learning model to the webpage, I used Flask framework, in which I routed the html's content and retrieved the required information with the help of html forms and processed the retrived data to the ML models. I saved the ML models using the pickle library and loaded them into the python code.
The data then processed through these models and the predicted value is displayed into the website using the jinja template.
