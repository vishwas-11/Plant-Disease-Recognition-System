import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction

def model_prediction(test_image):
    model = tf.keras.models.load_model('training_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home", "About", "Disease Recognition"])

#Home Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Page
if(app_mode=="About"):
    st.header("About")
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
        #### Content
        1. Train (70295 images)
        2. Valid (17572 images)        
        3. Test (33 images)
            
""")    
    
#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image")
    if(st.button("Show Image")):
        st.image(test_image,use_container_width=True)
    #Predict Image
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        plant_disease_solutions = {'Apple___Apple_scab': 'Apply fungicides during the early stages of growth. Prune infected branches and ensure proper spacing to increase airflow and reduce moisture.', 'Apple___Black_rot': 'Remove and destroy infected fruits, leaves, and branches. Use copper-based fungicides during the growing season to prevent infection.', 'Apple___Cedar_apple_rust': 'Remove nearby cedar trees if possible. Apply fungicides at the pre-bloom and post-bloom stages of apple trees.', 'Apple___healthy': 'Maintain good cultural practices such as proper pruning, fertilization, and irrigation to keep the plant healthy.', 'Blueberry___healthy': 'Provide proper irrigation, adequate drainage, and timely pruning. Use balanced fertilizers for nutrient support.', 'Cherry_(including_sour)___Powdery_mildew': 'Apply sulfur or other approved fungicides. Prune infected branches and increase spacing to improve air circulation.', 'Cherry_(including_sour)___healthy': 'Follow good cultural practices like pruning, fertilization, and irrigation. Avoid overhead watering to prevent infections.', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Apply fungicides during the early stages of the disease. Rotate crops annually and ensure proper spacing to improve air circulation.', 'Corn_(maize)___Common_rust_': 'Use rust-resistant corn varieties. Apply fungicides when signs of infection are detected, and ensure proper air circulation in the field.', 'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant corn varieties. Rotate crops, remove infected debris, and apply fungicides as needed.', 'Corn_(maize)___healthy': 'Ensure proper irrigation, soil health, and fertilization. Avoid overcrowding and ensure proper plant spacing.', 'Grape___Black_rot': 'Apply fungicides before and after bloom. Remove and destroy infected fruits and leaves. Use proper vine pruning and sanitation methods.', 'Grape___Esca_(Black_Measles)': 'Remove infected canes and prune dead wood. Apply fungicides at early infection stages. Avoid mechanical injuries to the vine.', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply copper-based fungicides. Remove and destroy infected leaves and debris to reduce the spread of the disease.', 'Grape___healthy': 'Maintain proper vine pruning, ensure sufficient air circulation, and avoid over-irrigation or waterlogging.', 'Orange___Haunglongbing_(Citrus_greening)': 'Use certified disease-free planting material. Remove and destroy infected trees. Control insect vectors (like psyllids) using insecticides.', 'Peach___Bacterial_spot': 'Apply copper-based bactericides during the early stages of infection. Prune infected branches and ensure proper air circulation.', 'Peach___healthy': 'Maintain proper pruning, irrigation, and fertilization. Regularly inspect for signs of bacterial spot or pest infestations.', 'Pepper,_bell___Bacterial_spot': 'Apply copper-based bactericides. Remove infected plants, ensure proper plant spacing, and avoid overhead watering.', 'Pepper,_bell___healthy': 'Ensure proper drainage, soil health, and balanced fertilization. Avoid overhead watering to prevent bacterial diseases.', 'Potato___Early_blight': 'Apply fungicides at the early stages of infection. Rotate crops and remove infected plant debris after harvest.', 'Potato___Late_blight': 'Apply fungicides with active ingredients like mancozeb or chlorothalonil. Use certified disease-free seeds and maintain proper field hygiene.', 'Potato___healthy': 'Plant certified disease-free seeds and maintain good field sanitation. Rotate crops annually to reduce disease pressure.', 'Raspberry___healthy': 'Maintain proper plant pruning, fertilization, and irrigation. Monitor regularly for signs of disease or pests.', 'Soybean___healthy': 'Follow good farming practices, crop rotation, and soil management. Use resistant varieties and provide optimal irrigation.', 'Squash___Powdery_mildew': 'Apply sulfur or other fungicides approved for powdery mildew. Prune infected leaves and increase spacing for better air circulation.', 'Strawberry___Leaf_scorch': 'Remove infected leaves and increase air circulation. Apply copper-based fungicides if the infection is severe.', 'Strawberry___healthy': 'Use healthy, disease-free plants. Provide proper drainage, irrigation, and plant spacing to avoid leaf scorch issues.', 'Tomato___Bacterial_spot': 'Apply copper-based bactericides. Remove infected leaves and avoid overhead watering to reduce moisture on leaves.', 'Tomato___Early_blight': 'Apply fungicides like chlorothalonil or mancozeb. Rotate crops and use resistant varieties to minimize infection risk.', 'Tomato___Late_blight': 'Apply fungicides at the first sign of infection. Remove and destroy infected plants and avoid overhead watering.', 'Tomato___Leaf_Mold': 'Apply fungicides and avoid overcrowding of plants. Provide good air circulation to reduce humidity levels around the leaves.', 'Tomato___Septoria_leaf_spot': 'Remove infected leaves and apply fungicides. Ensure adequate spacing and avoid overhead watering.', 'Tomato___Spider_mites Two-spotted_spider_mite': 'Use miticides or natural predators like ladybugs. Wash the leaves with water to dislodge mites and increase humidity.', 'Tomato___Target_Spot': 'Apply fungicides containing chlorothalonil. Remove infected plant debris and maintain good air circulation.', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Use resistant tomato varieties and control whitefly populations with insecticides. Remove and destroy infected plants.', 'Tomato___Tomato_mosaic_virus': 'Remove and destroy infected plants. Use resistant seeds and ensure proper field hygiene and crop rotation.', 'Tomato___healthy': 'Ensure proper irrigation, soil health, and fertilization. Remove weeds, ensure good airflow, and regularly inspect for early signs of disease.'}
        plant_disease_symptoms = {'Apple___Apple_scab': ['Olive green spots on leaves', 'Velvety texture on lesions', 'Premature leaf drop'], 'Apple___Black_rot': ['Dark, sunken lesions on fruit', 'Concentric rings on fruit lesions', 'Cankers on branches'], 'Apple___Cedar_apple_rust': ['Orange or rust-colored spots on leaves', 'Galls on cedar trees', 'Defoliation in severe cases'], 'Apple___healthy': ['Bright green leaves', 'No visible signs of infection', 'Healthy growth and apples'], 'Blueberry___healthy': ['Bright green leaves', 'No visible signs of infection', 'Healthy growth and berries'], 'Cherry_(including_sour)___Powdery_mildew': ['White powdery growth on leaves and stems', 'Leaf curling', 'Premature leaf drop'], 'Cherry_(including_sour)___healthy': ['Green leaves', 'No powdery residues', 'Healthy fruits'], 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': ['Gray or brown rectangular lesions on leaves', 'Yellowing around lesions', 'Reduced plant vigor'], 'Corn_(maize)___Common_rust_': ['Reddish-brown pustules on leaves', 'Yellowing around pustules', 'Leaves may tear'], 'Corn_(maize)___Northern_Leaf_Blight': ['Long, grayish-brown lesions on leaves', 'Lesions with smooth margins', 'Premature plant death in severe cases'], 'Corn_(maize)___healthy': ['Vivid green leaves', 'No visible lesions or rust', 'Healthy growth'], 'Grape___Black_rot': ['Dark, round lesions on leaves', 'Mummified berries', 'Defoliation in severe cases'], 'Grape___Esca_(Black_Measles)': ['Dark streaks or spots on berries', 'Yellowing of leaves', 'Dieback of vines'], 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': ['Brown or reddish spots on leaves', 'Yellowing and defoliation', 'Reduced fruit production'], 'Grape___healthy': ['Green leaves', 'No visible spots or lesions', 'Healthy grape clusters'], 'Orange___Haunglongbing_(Citrus_greening)': ['Yellow mottling of leaves', 'Misshapen, bitter fruits', 'Stunted tree growth'], 'Peach___Bacterial_spot': ['Small, water-soaked lesions on leaves', 'Dark spots on fruits', 'Leaf yellowing and drop'], 'Peach___healthy': ['Bright green leaves', 'Healthy, smooth fruits', 'No visible spots'], 'Pepper,_bell___Bacterial_spot': ['Small, dark, water-soaked spots on leaves', 'Spots enlarge and become brown', 'Leaf yellowing and drop'], 'Pepper,_bell___healthy': ['Green, healthy leaves', 'No spots on fruits', 'Healthy growth'], 'Potato___Early_blight': ['Dark, concentric spots on leaves', 'Yellowing around lesions', 'Reduced tuber growth'], 'Potato___Late_blight': ['Water-soaked, dark spots on leaves', 'Lesions with white fungal growth', 'Tubers with rot'], 'Potato___healthy': ['Vivid green leaves', 'Healthy tubers', 'No visible spots or lesions'], 'Raspberry___healthy': ['Bright green leaves', 'No visible spots or infections', 'Healthy fruit clusters'], 'Soybean___healthy': ['Healthy green leaves', 'No signs of rust or spots', 'Robust growth'], 'Squash___Powdery_mildew': ['White, powdery growth on leaves', 'Leaf curling', 'Premature leaf drop'], 'Strawberry___Leaf_scorch': ['Dark spots on leaves', 'Yellowing and drying of leaves', 'Reduced fruit yield'], 'Strawberry___healthy': ['Green leaves', 'No visible spots or discoloration', 'Healthy fruits'], 'Tomato___Bacterial_spot': ['Small, water-soaked spots on leaves', 'Dark spots on fruits', 'Leaf yellowing and drop'], 'Tomato___Early_blight': ['Dark, concentric spots on leaves', 'Yellowing around spots', 'Reduced fruit yield'], 'Tomato___Late_blight': ['Dark, water-soaked lesions on leaves', 'Lesions with white fungal growth', 'Fruits with rot'], 'Tomato___Leaf_Mold': ['Yellow spots on upper leaf surface', 'Grayish mold on leaf undersides', 'Leaf curling and drop'], 'Tomato___Septoria_leaf_spot': ['Small, circular spots on leaves', 'Spots with dark brown borders', 'Premature leaf drop'], 'Tomato___Spider_mites Two-spotted_spider_mite': ['Yellowing or bronzing of leaves', 'Webbing on undersides of leaves', 'Leaf drop in severe cases'], 'Tomato___Target_Spot': ['Dark, sunken lesions on leaves', 'Yellowing around lesions', 'Defoliation in severe cases'], 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': ['Yellowing and curling of leaves', 'Stunted growth', 'Reduced fruit yield'], 'Tomato___Tomato_mosaic_virus': ['Mosaic-like discoloration on leaves', 'Leaf curling and distortion', 'Reduced fruit yield'], 'Tomato___healthy': ['Green, healthy leaves', 'No visible spots or discoloration', 'Healthy fruits']}

        class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
        disease = class_name[result_index]
        symptoms = plant_disease_symptoms[disease]
        solution = plant_disease_solutions[disease]
        st.success("Model is Predicting it's a {}".format(disease))
        st.write("Symptoms: ")
        for i in range(len(symptoms)):
            st.success(f"{i+1}. {symptoms[i]}")
        st.write("Solution: ")
        st.success(solution)
        st.balloons()