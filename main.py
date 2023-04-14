from pathlib import Path
import streamlit as st
from PIL import Image
import base64

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
resume_file = current_dir / "assets" / "basels_cv.pdf"
profile_pic = current_dir / "assets" / "my_face4.png"

# --- GENERAL SETTINGS ---
PAGE_TITLE = "Digital CV | Basel Mather"
PAGE_ICON = "📝"
NAME = "Basel Mather"

DESCRIPTION = """
Data Science student at the University of Jordan, passion for learning. I aim to keep learning and become a better version of myself every day.
"""
EMAIL = "baselmathar@gmail.com"

SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/basel-mather/",
    "GitHub": "https://github.com/baselhusam",
    "Kaggle": "https://www.kaggle.com/baselmather"
    }


st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://raw.githubusercontent.com/baselhusam/Linear-Algebra/main/bg_2.jpg");
background-size: 100%;
background-position: top center;
background-repeat: y-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

# st.markdown(page_bg_img, unsafe_allow_html=True)


pf = """
<div class="effect3d"> Portfolio </div>
<style>
.effect3d { animation-name: effect3d }
@keyframes effect3d {
    to {
        text-shadow: 0 1px 0 #ccc, 0 2px 0 #c9c9c9, 0 3px 0 #bbb, 0 4px 0 #b9b9b9, 0 5px 0 #aaa, 0 6px 1px rgba(0, 0, 0, .1), 0 0 5px rgba(0, 0, 0, .1), 0 1px 3px rgba(0, 0, 0, .3), 0 3px 5px rgba(0, 0, 0, .2), 0 5px 10px rgba(0, 0, 0, .25), 0 10px 10px rgba(0, 0, 0, .2), 0 20px 20px rgba(0, 0, 0, .15)
    }
}
</style>
"""

# st.markdown(pf, unsafe_allow_html=True)


# --- LOAD CSS, PDF & PROFIL PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
with open(resume_file, "rb") as pdf_file:
    PDFbyte = pdf_file.read()
profile_pic = Image.open(profile_pic)


# --- HERO SECTION ---
col1, col2 = st.columns([1,1], gap="small")
with col1:
    # st.write('\n')
    st.write('\n')
    st.write('\n')
    st.image(profile_pic, width=290)

with col2:
    st.title(NAME)
    st.write(DESCRIPTION)
    st.download_button(
        label=" 📄 Download Resume",
        data=PDFbyte,
        file_name=resume_file.name,
        mime="application/octet-stream",
        )
    st.write("📫", EMAIL)


    # --- SOCIAL LINKS ---
    st.write('\n')
    cols = st.columns(len(SOCIAL_MEDIA))
    for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()): 
        cols[index].markdown(f"<a href='{link}'>{platform}</a> </div>", unsafe_allow_html=True)

        # cols[index].markdown(f"<div style='text-align: center;'> <a href='{link}'>{platform}</a> </div>", unsafe_allow_html=True) 



st.write('\n')
st.write('\n')
st.write('\n')

tab_titles = ["🔎 Overview", "🎢 Experience",  "🎯 Projects", "🎖️ Certifications", "🤝 Voluteering"]
tabs = st.tabs([s.center(15,"\u2001") for s in tab_titles])

# Overview
with tabs[0]:

    # --- About Me ---
    st.write('\n')
    st.header("About Me")
    st.write("---")
    st.markdown("""
    Data Science Student at the University of Jordan, passion for learning. 
    I've took many certificates in courses that related to my major, I like to avail my time. 
    I am also a productive person who can work efficient, I aim to keep learning and become a better version of myself every day.

    """)

    # --- EXPERIENCE & QUALIFICATIONS ---
    st.write('\n')
    st.header("Education")
    st.write("---")
    st.markdown("#### 🎓 University of Jordan  󠁪󠁪 󠁪󠁪 󠁪󠁪 󠁪󠁪 󠁪󠁪  󠁪󠁪 󠁪󠁪 |  󠁪󠁪 󠁪󠁪 󠁪󠁪 󠁪󠁪 󠁪󠁪 󠁪󠁪   2020 - 2024", unsafe_allow_html=True)
    st.markdown(
        """-  󠁪󠁪  󠁪󠁪  󠁪󠁪  Majoring in Data Science  󠁪󠁪 󠁪󠁪 󠁪󠁪 -  󠁪󠁪 󠁪󠁪 󠁪󠁪 **GPA:  󠁪󠁪 3.8 / 4.0**
    """, unsafe_allow_html=True
    )


    # --- SKILLS ---
    st.write('\n')
    st.write('\n')
    st.header("Skills")
    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hard Skills")
        
        st.write(
            """
        1.  Programming: Python (Numpy, Pandas, Matplotlib, Seaborn, Scikit-learn, Tensorflow)
        2.  Data Egineering: ETL, Data Cleaning, Data Wrangling, Data Visualization
        3.  Data Analysis: Exploratory Data Analysis, Data Mining, Data Modeling, Data Visualization 
        4.  Mathematics for Machine Learning: Linear Algebra, Multivariate Calculus, Probability, Statistics
        5.  Machine Learning: Supervised Learning, Unsupervised Learning, Reinforcement Learning
        6.  Deep Learning: Convolutional Neural Networks, Recurrent Neural Networks
        7.  Computer Vision: Image Processing, Object Detection, Image Recognition
        8.  Natural Language Processing: Text Processing, Text Classification
        9.  TensorFlow: Keras, Tensorflow Hub, Tensorflow Datasets
        10. SQL Databases: Postgres, MySQL
        11. Git & GitHub
        12. Problem Solving
        """
        )

    with col2:
        st.subheader("Soft Skills")
        st.write(
            """
        1. Graphic Design
        2. Motion Graphics
        3. Audio Engineering & Music Production
        4.  Communication
        5.  Teamwork
        6.  Leadership
        7.  Time Management
        8.  Creativity
        9.  Word Under Pressure
        """
        )

# Experience
with tabs[1]:

    st.write('\n')
    st.write('\n')
    st.header("🧑‍💻 Experience")
    st.write("---")

    # --- JOB 1
    st.markdown(" ### **Data Science Intern | SHAI FOR AI**")
    st.write("09/2022 - 12/2022")
    st.markdown(
        """
        I completed a three-month internship at SHAI FOR AI, where I gained valuable experience in data science and software engineering. 
        During my internship, I worked on several huge projects from September 10th to December 10th. My responsibilities included 
        completing projects on time and utilizing my technical and soft skills to deliver high-quality work.

        <br> 

        One of my main projects was the Playing Card Detection & Tarneeb game, a computer vision project that involved detecting playing 
        cards and building a Tarneeb game based on the model detections. I also worked on a search engine project using NLP and a PUBG 
        Finish Placement Prediction project using machine learning, deep learning, and data analysis.

        <br>

        Additionally, I was part of a team that built a Library project using software engineering, OOP, clean code, and other programming 
        concepts. This project involved writing functions, comments, and descriptions, sharing work with others, uploading to GitHub, 
        making commits, and working in a team.

        <br>

        Throughout my internship, I used Jupyter Notebook and VS Code as software and wrote code in Python. After each project, I made 
        presentations to discuss the results and process, and improve my soft skills. The atmosphere at SHAI FOR AI was flexible and 
        supportive, and I had the opportunity to interact with amazing people who helped me grow as a 
        data scientist and software engineer.


        <br>

         """, unsafe_allow_html=True)

    displayPDF("assets/shai_for_ai.pdf")

    

# Projects & Accomplishments 
with tabs[2]:

    st.write('\n')
    st.markdown("## 🎯 Projects & Accomplishments")


    st.write('\n')
    st.markdown("#### 🎶 Song Popularity Prediction web app", unsafe_allow_html=True)
    with st.expander("Song Popularity Prediction web app"):
        st.markdown(
            """ 
                <br>

                #### 📝 Description
                ---
                Building a ML Model to predict the popularity of a song based on its audio features. <br>
                Then building a web app using Streamlit to deploy the model and make it available for everyone to use. 
                
                <br>


                """, unsafe_allow_html=True)

        video_file = open('assets/spp-web-app.mp4', 'rb')
        video_bytes_spp = video_file.read()

        st.video(video_bytes_spp)
        
        st.markdown(
            """
                <br> 

                #### 🔗 Links 
                ---
                👉 The web app is [available here](https://baselhusam-song-s-popularity-prediction-application-main-7nbbe9.streamlit.app/). <br> <br>
                👉 The source code for the web app is [available here](https://github.com/baselhusam/Song-s-Popularity-Prediction-Applicationhttps://github.com/baselhusam/Song-Popularity-Prediction). <br> <br>
                👉 the source code for building the model is [available here](https://github.com/baselhusam/Song-Popularity-Prediction). <br> <br>
                """, unsafe_allow_html=True)
        
    st.write('\n')
    st.markdown("#### 🃏 Playing Card Detection & Tarneeb Game", unsafe_allow_html=True)
    with st.expander("Playing Card Detection & Tarneeb Game"):
        st.markdown(
            """ 
                <br>

                #### 📝 Description
                ---

                This project is a computer vision project that contains two phases. 
                
                <br>

                ###### Phase one: Playing Cards Detection
                This phase is about building a model to detect the playing cards and count them with their classes.

                <br>

                ###### Phase two: Tarneeb Game ( The Video Below )
                In this phase, we had to build the Tarneeb game and know the winner (note that in the video the tarneeb is the heart (H) which is why the fourth player is the winner).

                <br>
                """, unsafe_allow_html=True)       

        video_file = open('assets/pcd-tg.mp4', 'rb')
        video_bytes_pcd = video_file.read()

        st.video(video_bytes_pcd)

        st.markdown("""
                <br>

                #### 🔗 Links
                ---
                👉 The source code for this project is [available here](https://github.com/baselhusam/Playing-Cards-Detection-with-Tarneeb). 

        """, unsafe_allow_html=True)
    
    st.write('\n')
    st.markdown("#### 🌐 Search Engine ", unsafe_allow_html=True)
    with st.expander("Search Engine"):
        st.markdown(
            """ 
                <br>

                #### 📝 Description
                ---
                Building a search engine from raw data that contain of 500 Arabic articels from Husana website. <br>

                #### 🔗 Links
                ---
                👉 The source code for this project is [available here]().

            """, unsafe_allow_html=True)
    
    st.write('\n')
    st.markdown("#### 👤 Blurify.AI", unsafe_allow_html=True)
    with st.expander("Blurify.AI", ):
        st.markdown(""" 

                    <br> 
                    
                    #### 📝 Description
                    ---

                    ##### 🤔 What is Blurify.AI?
                    Blurify.AI is a web-based application that allows you to upload an image and detect and blur all faces within that image. 
                    The project was built using the YOLO v8 and HaarCascade Classifier object detection algorithms and Streamlit framework. 
                    The application can be used to anonymize faces in images, preserving privacy and confidentiality. It can also be used 
                    for creative purposes such as adding a blur effect to a portrait.

                    <br>

                    ##### 🔨 How was it built?

                    Blurify.AI was built using the YOLO v8 and HaarCascade Classifier object detection algorithms and Streamlit web framework. The YOLO v8 algorithm is a state-of-the-art object detection algorithm that is highly accurate and efficient. Streamlit is a popular web framework that allows you to quickly build and deploy web-based applications using Python. The project was trained on a dataset of faces to detect and blur faces in images uploaded by the user.

                    <br>
                """, unsafe_allow_html=True)
        
        video_file = open('assets/blurify.mp4', 'rb')
        video_bytes_blurify = video_file.read()

        st.video(video_bytes_blurify)

        st.markdown("""
                    #### 🔗 Links
                    ---
                    👉 The Source code for the web application is [available here](https://github.com/baselhusam/Blurify.AI). <br> <br>
                    👉 Ckech the GitHub repo for building the models, evaluations, etc. the [link](https://github.com/baselhusam/Face-Blurring).

        """, unsafe_allow_html=True)
        
    st.write('\n')
    st.markdown("#### 😶‍🌫 Face Blurring", unsafe_allow_html=True)
    with st.expander("Face Blurring"):
        st.markdown("""

                    <br>

                    #### 📝 Description
                    ---
                    This project aims to blur faces in images while keeping the rest of the image unchanged. To achieve this, we utilized two approaches to detect faces in an image. <br>

                    The solution consists of two models: A pre-trained model using Haar Cascade Classifier , and a trained model from scratch with YOLO v8.

                    <br>

                    #### 🔗 Links
                    ---
                    👉 The Source code for the project is [available here](https://github.com/baselhusam/Face-Blurring). 

        """, unsafe_allow_html=True)
        
    st.write('\n')
    st.markdown("#### 🎮 PUBG Finish Placement Prediction", unsafe_allow_html=True)
    with st.expander("PUBG Finish Placement Prediction"):
        st.write(""" 

                <br>

                #### 📝 Description
                ---
                This project aims to predict the finish placement of a player in a PUBG match based on the player's final stats, on a scale of 1 to 4. <br> <br>
                This project is a part of the [Kaggle PUBG Finish Placement Prediction](https://www.kaggle.com/c/pubg-finish-placement-prediction) competition. <br> <br>
                The solution has been built using Machine Learning with the XGBoost algorithm. <br>

                #### 🔗 Links
                ---

                👉 The Source code for the project is [available here](https://github.com/baselhusam/PUBG-Finish-Placement-Prediction-Project). 
                
                """, unsafe_allow_html=True)
        
    st.write('\n')
    st.markdown("#### 😷 Face Mask Classifier", unsafe_allow_html=True)
    with st.expander("Face Mask Classification"):
        st.markdown("""

                <br>

                #### 📝 Description
                ---
                This problem is designed to detect whether a person in an image is wearing a face mask or not. 
                It is based on deep learning models trained on the Face Mask Classification dataset from Kaggle. <br>

                This solution consists of two models: Transfer Learning with the ResNet50 model and a custom model built 
                using TensorFlow. Both models trained on the same dataset have achieved high accuracy in detecting 
                whether a person is wearing a mask. <br>



                #### 🔗 Links
                ---
                👉 The Source code for the project is [available here]().
                """, unsafe_allow_html=True)

# Certifications
with tabs[3]:

    st.write('\n')
    st.markdown("## Certifications 🎖️")
    
    st.write('\n')
    st.markdown("##### Mathematics for Machine Learning Specialization | Coursera - Imperial College")
    with st.expander("Mathematics for Machine Learning Specialization | Coursera - Imperial College London"):

        # displayPDF(r"D:\Projects\Porfolio\assets\certifications\LinkedIn\CertificateOfCompletion_Data Fluency Exploring and Describing Data (1).pdf")
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//Mathematics for ML//math_ml_cert.jpg")
            st.image(img,  width=700, use_column_width='always') 

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.
        It consists of three courses: Linear Algebra, Multivariate Calculus, and PCA. """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h5 align='center'> Linear Algebra </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//Mathematics for ML//la_cert.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col2:
            st.markdown("<h5 align='center'> Multivariate Calculus </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//Mathematics for ML//mult_var_cal_cert.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col3:
            st.markdown("<h5 align='center'> PCA </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//Mathematics for ML//pca_cert.jpg")
            st.image(img,  width=200, use_column_width='always')

    st.write('\n')
    st.markdown("##### Machine Learning Specialization | Coursera - DeepLearning.AI ")
    with st.expander("Machine Learning Specialization | Coursera - DeepLearning.AI & Stanford University"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//Machine Learning Specialization//ml_cert.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h5 align='center'> Supervised Machine Learning </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//Machine Learning Specialization//sup_cert.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col2:
            st.markdown("<h5 align='center'> Advanced Learning Algorithm <br> </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//Machine Learning Specialization//adv_cert.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col3:
            st.markdown("<h5 align='center'> Unsupervised, Reinforcement Learning </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//Machine Learning Specialization//uns_cert.jpg")
            st.image(img,  width=200, use_column_width='always')

    st.write('\n')
    st.markdown("##### Deep Learning Specialization | Coursera - DeepLearning.AI")
    with st.expander("Deep Learning Specialization | Coursera - DeepLearning.AI & Stanford University"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//Deep Learning Specialization//dl_cert.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h5 align='center'> Neural Networks and Deep Learning </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//Deep Learning Specialization//nn_cert.jpg")
            st.image(img,  width=200, use_column_width='always')

            st.write('\n')

            st.markdown("<h5 align='center'> Convolutional Neural Network </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//Deep Learning Specialization//cnn_cert.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col2:
            st.markdown("<h5 align='center'> Improving Deep Neural Networks </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//Deep Learning Specialization//hyp_cert.jpg")
            st.image(img,  width=200, use_column_width='always')

            st.write('\n')

            st.markdown("<h5 align='center'> Sequence Models <br> <br> </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//Deep Learning Specialization//seq_cert.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col3:
            st.markdown("<h5 align='center'> Structuring Machine Learning Projects </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//Deep Learning Specialization//ml_project_cert.jpg")
            st.image(img,  width=200, use_column_width='always')

            st.write('\n')

            st.markdown("<h5 align='center'> Deep Learning Badge <br> <br> </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//Deep Learning Specialization//dl_badge.jpg")
            st.image(img,  width=200, use_column_width='always')

    st.write('\n')
    st.markdown("##### TensorFlow Developer Certificate | Coursera - DeepLearning.AI ")
    with st.expander("TensorFlow Developer Certificate | Coursera - DeepLearning.AI "):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//TensorFlow Professional Certificate//tf.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h5 align='center'> Introduction TensorFlow for AI, ML, and DL </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//TensorFlow Professional Certificate//tf1.jpg")
            st.image(img,  width=200, use_column_width='always')

            st.write('\n')

            st.markdown("<h5 align='center'> Sequence, Time Series, and Prediction </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//TensorFlow Professional Certificate//tf4.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col2:
            st.markdown("<h5 align='center'> Convolutional Neural Networks in TensorFlow </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//TensorFlow Professional Certificate//tf2.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col3:
            st.markdown("<h5 align='center'> NLP in TensorFlow <br> <br> </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//TensorFlow Professional Certificate//tf3.jpg")
            st.image(img,  width=200, use_column_width='always')

    st.write('\n')
    st.markdown("##### TensorFlow: Data and Deployment Specializatoin | Coursera - DeepLearning.AI")
    with st.expander("TensorFlow: Data and Deployment Specializatoin | Coursera - DeepLearning.AI"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//TensorFlow.js//tf_dep.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h5 align='center'> Browser-based Models with TF.js </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//TensorFlow.js//tf_dep1.jpg")
            st.image(img,  width=200, use_column_width='always')

            st.write('\n')

            st.markdown("<h5 align='center'> Advanced Deployment Scenarios with TF </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//TensorFlow.js//tf_dep4.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col2:
            st.markdown("<h5 align='center'> Device Based Models with TF Lite </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//TensorFlow.js//tf_dep3.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col3:
            st.markdown("<h5 align='center'> Data Pipelines with TF Data Services </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//TensorFlow.js//tf_dep3.jpg")
            st.image(img,  width=200, use_column_width='always')

    st.write('\n')
    st.markdown("##### Natural Language Processing Specialization | Coursera - DeepLearning.AI")
    with st.expander("Natural Language Processing Specialization | Coursera - DeepLearning.AI"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//NLP Specialization//nlp.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h5 align='center'> NLP with Classification and Vector Spaces </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//NLP Specialization//nlp1.jpg")
            st.image(img,  width=200, use_column_width='always')

            st.write('\n')

            st.markdown("<h5 align='center'> NLP with Attention Models </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//NLP Specialization//nlp4.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col2:
            st.markdown("<h5 align='center'> NLP with Probabilistic Models </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//NLP Specialization//nlp2.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col3:
            st.markdown("<h5 align='center'> NLP with Sequence Models </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//NLP Specialization//nlp3.jpg")
            st.image(img,  width=200, use_column_width='always')

    st.write('\n')
    st.markdown("##### TensorFlow: Data and Deployment Specializatoin | Coursera - DeepLearning.AI")
    with st.expander("AI For Medicine Specialization | Coursera - DeepLearning.AI"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//AI for Medicine//ai_med.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h5 align='center'> AI for Medical Diagnosis </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//AI for Medicine//ai_med1.jpg")
            st.image(img,  width=200, use_column_width='always')


        with col2:
            st.markdown("<h5 align='center'> AI for Medical Treatment </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//AI for Medicine//ai_med2.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col3:
            st.markdown("<h5 align='center'> AI for Medical Prognosis </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//AI for Medicine//ai_med3.jpg")
            st.image(img,  width=200, use_column_width='always')

    st.write('\n')
    st.markdown("##### AWS Fundementals Specialization | Coursera - AWS")
    with st.expander("AWS Fundementals Specialization | Coursera - AWS"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//AWS Specialization//aws.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h5 align='center'> AWS Fundamentals: Going Cloud-Native </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//AWS Specialization//aws1.jpg")
            st.image(img,  width=200, use_column_width='always')

            st.write('\n')

            st.markdown("<h5 align='center'> AWS Fundamentals: Building Serverless Apps </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//AWS Specialization//aws4.jpg")
            st.image(img,  width=200, use_column_width='always')


        with col2:
            st.markdown("<h5 align='center'> AWS Fundamentals: Addressing Security Risk </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//AWS Specialization//aws2.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col3:
            st.markdown("<h5 align='center'> AWS Fundamentals: Migrating to the Cloud </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//AWS Specialization//aws3.jpg")
            st.image(img,  width=200, use_column_width='always')

    st.write('\n')
    st.markdown("##### Data Visualization with Python | Coursera - IBM")
    with st.expander("Data Visualization with Python | Coursera - IBM"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//data_vis.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

    st.write('\n')
    st.markdown("##### Machine Learning with Python | Coursera - IBM")
    with st.expander("Machine Learning with Python | Coursera - IBM"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//ml_python.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

    st.write('\n')
    st.markdown("##### Python Project for Data Engineering | Coursera - IBM")
    with st.expander("Python Project for Data Engineering | Coursera - IBM"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//data_eng_proj.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

    st.write('\n')
    st.markdown("##### Data Cleaning & Preprocessing with Pandas | 365 days of Data Science")
    with st.expander("Data Cleaning & Preprocessing with Pandas | 365 days of Data Science"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//365 Data Science//data_clean.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

    st.write('\n')
    st.markdown("##### AWS Machine Learning Foundation | Udacity - AWS")
    with st.expander("AWS Machine Learning Foundation | Udacity - AWS"): 
        st.write('\n')

        col1, col2 = st.columns(2)

        with col1:

            cola, colb, colc = st.columns([0.1,1,0.1])
            with colb:
                img = Image.open(".//assets//certifications//Udacity//AWS_ML.jpg")
                st.image(img,  width=700, use_column_width='always')

        with col2:

            colx, coly, colz = st.columns([0.1,1,0.1])
            with coly:
                img = Image.open(".//assets//certifications//Udacity//AWS-Course-badge-Gold.png")
                st.image(img,  width=600, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

    st.write('\n')
    st.markdown("##### Data Analysis Track | Udacity")
    with st.expander("Data Analysis Track | Udacity"):
        st.write('\n')

        col1, col2,col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Udacity//DataAnalysis__Udacity.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

    st.write('\n')
    st.markdown("##### Data Engineering For Everyone | Udacity")
    with st.expander("Data Engineering For Everyone | DataCamp"):
        st.write('\n')

        col1, col2,col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Data camp//data_eng.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

    st.write('\n')
    st.markdown("##### Introduction to Python | DataCamp")
    with st.expander("Introduction to Python | DataCamp"):
        st.write('\n')

        col1, col2,col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Data camp//python.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

    st.write('\n')
    st.markdown("##### LinkedIn Certificates")
    with st.expander("LinkedIn Certificates"):
        st.write('\n')

        col1, col2,col3 = st.columns([1,1,1])

        with col1:
            st.markdown("<h5 align='center'> Learning Excel: Data Analysis </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//LinkedIn//l1.jpg")
            st.image(img,  width=700, use_column_width='always')

            st.write('\n')

            st.markdown("<h5 align='center'> Excel Statistics Essential Training 1 </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//LinkedIn//l4.jpg")
            st.image(img,  width=700, use_column_width='always')

        with col2:
            st.markdown("<h5 align='center'> Learning Data Analysis 1 <br> <br> </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//LinkedIn//l2.jpg")
            st.image(img,  width=700, use_column_width='always')

            st.write('\n')

            st.markdown("<h5 align='center'> Data Fluency: Exploring & Describing Data </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//LinkedIn//l5.jpg")
            st.image(img,  width=700, use_column_width='always')

        with col3:
            st.markdown("<h5 align='center'> Learning Data Analysis 2 <br> <br> </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//LinkedIn//l3.jpg")
            st.image(img,  width=700, use_column_width='always')

            st.write('\n')

            st.markdown("<h5 align='center'> Non-tech skills of Effictive Data Scientist </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//LinkedIn//l6.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

    st.write('\n')
    st.markdown("##### IEEE Extreme | IEEE")
    with st.expander("IEEE Extreme | IEEE"):
        st.write('\n')

        col1, col2,col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//IEEE//IEEE Extreme.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

    st.write('\n')
    st.markdown("##### Python Programming | Udemy")
    with st.expander("Python Programming | Udemy"):
        st.write('\n')

        col1, col2,col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Udemy//python.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" This Specialization covers the mathematical and statistical foundations of machine learning.""")

    st.write('\n')
    st.markdown("##### Macine Learning Intern | SteamCenter")
    with st.expander("Machine Learning Intern | SteamCenter"):
        st.write('\n')

        col1, col2,col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//steam_center.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write("07/2022 - 08/2022")
        st.markdown("""

        Machine Learning Workshop with 24 hours of training. We've started from zero to hero, 
        this workshop contains a well explained of machine learning and AI. After that, we went to 
        understand some machine learning algorithms, and each day we had to do tasks to ensure that 
        we understood everything well. Finally, We've worked on a machine learning project that aims 
        to predict the car price based on some features of cars, this project contains the steps for 
        every machine learning project and the fundamental steps such as Data Exploration, Data Cleaning, 
        Handling Missing Values, Encode Categorical Features, Try Different Machine Learning Algorithms, 
        Hyperparameter Tuning, Pick Performance Metric and Evaluate the model.
        

        <br>
        """, unsafe_allow_html=True)


# Volunteering
with tabs[4]:

    st.subheader("Volunteering 🤝")

    st.write('\n')
    st.markdown("""### Technical Team Lead at IEEE CIS""", unsafe_allow_html=True)
    with st.expander("See more"):
        st.write('\n')

        st.markdown("""
        
                    As a Technical Team Lead at the IEEE Computational Intelligence Society 
                    my role is to manage the technical side of this society, and the technical side
                    contains courses, workshops, guided projects, and competitions.

                    I have to make interviews with the applicants who want to give a course or a workshop, and 
                    I have to make sure that the course or the workshop is up to the IEEE CIS standards.

                    For now, I've given 1 course and 1 workshop that are related to Data Preparation.

                    <br>

                    #### Data Preparation Workshop
                    
                    In this workshop I've talked about the importance of data preparation, and I've talked about the 
                    different steps of data preparation, and I've talked about the different tools that we can use to
                    prepare our data. Also, We've applied the different steps of data preparation on a real dataset.

                    <br>

                    The GitHub repository of this workshop is [here](https://github.com/baselhusam/Titanic-Dataset---Data-Preparation).
                    <br> <br>

                    #### Data Preparation Course

                    In this course I've talked about the importance of data preparation, and I've went deep into the data preparation phases,
                    from the data collection phase to the splitting the data into training and testing sets. This course is a 15 hours course, with 
                    a final project that is related to data preparation, each student has to prepare a dataset and apply the different steps of data preparation
                    on his own project.

                    The GitHub repository of this course is [here](https://github.com/baselhusam/Data-Preparation-Course-IEEE-CIS).
                    """, unsafe_allow_html=True)
    

    
