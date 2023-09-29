from pathlib import Path
import streamlit as st
from PIL import Image
import base64

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
resume_file = current_dir / "assets" / "Basel CV.pdf"
profile_pic = current_dir / "assets" / "my_face4.png"

# --- GENERAL SETTINGS ---
PAGE_TITLE = "Portfolio | Basel Mather"
PAGE_ICON = "💼"
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

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

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

tab_titles = ["🔎 Overview", "🎢 Experience",  "🎯 Projects", "🎖️ Certifications", "🤝 Voluteering", "✉️ Recommendation"]
tabs = st.tabs([s.center(12,"\u2001") for s in tab_titles])

# Overview
with tabs[0]:

    # --- About Me ---
    st.write('\n')
    st.header("🙋‍♂️ About Me")
    st.write("---")
    st.markdown("""
    Data Science Student at the University of Jordan, passion for learning. 
    I've took many certificates in courses that related to my major, I like to avail my time. 
    I am also a productive person who can work efficient, I aim to keep learning and become a better version of myself every day.

    """)

    # --- EXPERIENCE & QUALIFICATIONS ---
    st.write('\n')
    st.header("🎓 Education")
    st.write("---")
    st.markdown("#### University of Jordan  󠁪󠁪 󠁪󠁪 󠁪󠁪 󠁪󠁪 󠁪󠁪  󠁪󠁪 󠁪󠁪 |  󠁪󠁪 󠁪󠁪 󠁪󠁪 󠁪󠁪 󠁪󠁪 󠁪󠁪   2020 - 2024", unsafe_allow_html=True)
    st.markdown(
        """-  󠁪󠁪  󠁪󠁪  󠁪󠁪  Majoring in Data Science  󠁪󠁪 󠁪󠁪 󠁪󠁪 -  󠁪󠁪 󠁪󠁪 󠁪󠁪 **GPA:  󠁪󠁪 3.8 / 4.0**
    """, unsafe_allow_html=True
    )


    # --- SKILLS ---
    st.write('\n')
    st.write('\n')
    st.header("💡 Skills")
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
    st.header("🧑‍💻 Experience")
    st.write("---")

    # --- JOB 1
    st.markdown(" ### **Machine Learning Engineer Intern | Nafith Logistics**")
    st.write("07/2023 - 9/2023 | Amman, Jordan | On-Site")
    st.markdown(
        """
        During my ML Engineer Internship at Nafith Logistics International, I gained valuable experience in various aspects of machine learning and computer vision. My responsibilities included:

        **Image Labeling and Dataset Preparation:** I meticulously labeled images of vehicle license plates and performed dataset preprocessing for model training.

        **Model Development:** I was involved in training models for license plate recognition and optical character recognition (OCR) to extract inner numbers/characters from license plates.

        **Inference and Post-Processing:** I successfully applied these models to perform image inference and developed OCR post-processing algorithms to convert OCR results into complete license plate strings.

        **Object Detection and Tracking:** I implemented object detection using classical approaches, which involved techniques like contour analysis and background removal. I also enhanced detection results through image pre/post-processing and integrated classical approaches with YOLO.

        **Tracking Algorithms:** I utilized the SORT algorithm for object tracking and collected relevant datasets for these tasks.

        **Infrastructure and Deployment:** I gained proficiency in setting up Docker environments, built Docker files for running license plate recognition scripts, and established GPU connectivity. Additionally, I contributed to deployment using Docker Compose.

        Throughout my internship, I acquired practical knowledge in CUDA installation, CuDNN, Ubuntu, and object counting using a classifier task, among other skills. This experience has enriched my understanding of computer vision, allowing me to contribute effectively to complex projects in these domains.
        
        **Skills:** Image Processing · YOLO · Inference · Image Labelling · Object Tracking · Docker · Data Collection · Ubuntu · Object Detection · Computer Vision

        <br> 

         """, unsafe_allow_html=True)

    # --- JOB 1
    st.markdown(" ### **Data Science Intern | SHAI FOR AI**")
    st.write("09/2022 - 12/2022 | Amman, Jordan | On-Site")
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


        <br> <br>

         """, unsafe_allow_html=True)

    img = Image.open(".//assets//shai_for_ai.jpg")
    st.image(img, width=700)

# Projects & Accomplishments 
with tabs[2]:

    st.write('\n')
    st.markdown("### 🧠 Machine & Deep Learning")
    st.markdown('---')

    st.markdown("##### 👉 ClickML " , unsafe_allow_html=True)
    with st.expander("ClickML Web Application"):
        st.write("\n")
        col1, col2, col3 = st.columns([0.25,1,0.25])
        col2.image(".//assets//ClickML-Logo.png")
        st.write("\n")
        st.write("ClickML is a powerful and user-friendly platform designed to empower non-technologists to create predictive models for their businesses effortlessly. With ClickML, you can build, train, evaluate, and fine-tune machine learning models with just a few clicks, eliminating the need for complex coding.")

        st.markdown("### 🎬 ClickML Promo")
        st.markdown('---')
        st.video(".//assets//ClickML-Promo.mp4")
        st.write("\n")

        st.markdown("### 📝 Project Description")
        st.markdown('---')
        st.write("ClickML aims to democratize machine learning by providing a no-code platform that simplifies the process of building predictive models. Our platform enables users to leverage the power of machine learning without having to dive into the intricacies of coding.")
        st.write("\n")

        st.markdown("### 🔍 ClickML Tutorial")
        st.markdown('---')
        st.video(".//assets//ClickML-Tutorial.mp4")
        st.write("\n")

        st.markdown("### 🔗 Links")
        st.markdown('---')
        st.markdown("""
                    
        👉 ClickML Web Application: [__available here__](https://clickml.streamlit.app/). <br>
        
        👉 ClickML Roadmap: [__available here__](https://clickml-roadmap.streamlit.app). <br>
        
        👉 ClickML Official LinkedIn Page: [__available here__](https://www.linkedin.com/company/clickml/?viewAsMember=true). <br>
        
        👉 ClickML Source Code in GitHub: [__available here__](https://github.com/baselhusam/clickml). <br>
        
                    """, unsafe_allow_html=True)


    st.write('\n')
    st.markdown("##### 🎶 Song Popularity Prediction web app ", unsafe_allow_html=True)
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
                👉 The web app is [__available here__](https://baselhusam-song-s-popularity-prediction-application-main-7nbbe9.streamlit.app/). <br> <br>
                👉 The source code for the web app is [__available here__](https://github.com/baselhusam/Song-s-Popularity-Prediction-Applicationhttps://github.com/baselhusam/Song-Popularity-Prediction). <br> <br>
                👉 the source code for building the model is [__available here__](https://github.com/baselhusam/Song-Popularity-Prediction). <br> <br>
                """, unsafe_allow_html=True)
    
    st.write('\n')
    st.markdown("##### 🎮 PUBG Finish Placement Prediction", unsafe_allow_html=True)
    with st.expander("PUBG Finish Placement Prediction"):
        st.write(""" 

                <br>

                #### 📝 Description
                ---
                This project aims to predict the finish placement of a player in a PUBG match based on the player's final stats, on a scale of 1 to 4. <br> <br>
                This project is a part of the [__Kaggle PUBG Finish Placement Prediction__](https://www.kaggle.com/c/pubg-finish-placement-prediction) competition. <br> <br>
                The solution has been built using Machine Learning with the XGBoost algorithm. <br>

                #### 🔗 Links
                ---

                👉 The Source code for the project is [__available here__](https://github.com/baselhusam/PUBG-Finish-Placement-Prediction-Project). 
                
                """, unsafe_allow_html=True)
        
    
    st.write('\n')
    st.write('\n')
    st.markdown("### 👁️‍🗨️ Computer Vision")
    st.markdown('---')

    st.markdown("##### 🎨 Art Styleify " , unsafe_allow_html=True)
    with st.expander("ClickML Web Application"):
        st.write("\n")
        col1, col2, col3 = st.columns([0.25,1,0.25])
        col2.image(".//assets//Art-Styleify-Logo.png")
        st.write("\n")
        st.write("Art Styleify is a web application that allows you to transform your photos into stunning masterpieces using various artistic styles. With just a few clicks, you can apply the style of famous artists like Van Gogh, Josef Sima, Man Ray, Max Ernst, and Wassily Kandinsky to your images. Unleash your creativity and create unique artwork effortlessly!")

        st.markdown("### 🎬 Art Styleify Promo")
        st.markdown('---')
        st.video(".//assets//Art-Styleify-Promo.mp4")
        st.write("\n")

    

        st.markdown("### 🔗 Links")
        st.markdown('---')
        st.markdown("""
                    
        👉 Art Styleify Web Application: [__available here__](https://art-styleify.streamlit.app/). <br>
        
        👉 Art Styleify Source Code:  [__available here__](https://github.com/baselhusam/art-styleify/) <br>
        
                    """, unsafe_allow_html=True)
        st.write("\n")

    st.write("\n")
    st.markdown("##### 🃏 Playing Card Detection & Tarneeb Game", unsafe_allow_html=True)
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
                In this phase, we had to build the Tarneeb game and know the winner (note that in the video the tarneeb is the heart 
                (H) which is why the fourth player is the winner).

                <br>
                """, unsafe_allow_html=True)       

        video_file = open('assets/pcd-tg.mp4', 'rb')
        video_bytes_pcd = video_file.read()

        st.video(video_bytes_pcd)

        st.markdown("""
                <br>

                #### 🔗 Links
                ---
                👉 The source code for this project is [__available here__](https://github.com/baselhusam/Playing-Cards-Detection-with-Tarneeb). 

        """, unsafe_allow_html=True)

    st.write('\n')      
    st.markdown("##### 😶‍🌫 Face Blurring", unsafe_allow_html=True)
    with st.expander("Face Blurring"):
        st.markdown("""

                    <br>

                    #### 📝 Description
                    ---
                    This project aims to blur faces in images while keeping the rest of the image unchanged. To achieve this, we utilized 
                    two approaches to detect faces in an image. <br>

                    The solution consists of two models: A pre-trained model using Haar Cascade Classifier , and a trained model 
                    from scratch with YOLO v8.

                    <br>

                    #### 🔗 Links
                    ---
                    👉 The Source code for the project is [__available here__](https://github.com/baselhusam/Face-Blurring). 

        """, unsafe_allow_html=True)

    st.write('\n')   
    st.markdown("##### 👤 Blurify.AI ", unsafe_allow_html=True)
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

                    Blurify.AI was built using the YOLO v8 and HaarCascade Classifier object detection algorithms and Streamlit web framework. 
                    The YOLO v8 algorithm is a state-of-the-art object detection algorithm that is highly accurate and efficient. 
                    Streamlit is a popular web framework that allows you to quickly build and deploy web-based applications using Python. 
                    The project was trained on a dataset of faces to detect and blur faces in images uploaded by the user.

                    <br>
                """, unsafe_allow_html=True)
        
        video_file = open('assets/blurify.mp4', 'rb')
        video_bytes_blurify = video_file.read()

        st.video(video_bytes_blurify)

        st.markdown("""
                    #### 🔗 Links
                    ---
                    👉 The Source code for the web application is [__available here__](https://github.com/baselhusam/Blurify.AI). <br> <br>
                    👉 Ckech the GitHub repo for building the models, evaluations, etc. [__the link__](https://github.com/baselhusam/Face-Blurring).

        """, unsafe_allow_html=True)

    st.write('\n')
    st.markdown("##### 😷 Face Mask Classifier", unsafe_allow_html=True)
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
                👉 The Source code for the project is [__available here__](https://github.com/baselhusam/Face-Mask-Detection).
                """, unsafe_allow_html=True)

    st.write('\n')
    st.write('\n')
    st.markdown("### 💬 Natural Language Processing")
    st.markdown('---')
    st.markdown("##### 🌐 Search Engine ", unsafe_allow_html=True)
    with st.expander("Search Engine"):
        st.markdown(
            """ 
                <br>

                #### 📝 Description
                ---
                Building a search engine from raw data that contain of 500 Arabic articels from Husana website. <br>
                This project has been built using the following steps: <br>
                1. we have to clean the data and remove the stop words. <br>
                2. we have to tokenize the data and build the inverted index. <br>
                3. we make the search engine with the cosine similarity by TF-IDF. <br>
                4. we have to build the user interface using Streamlit. 
                <br>


                """, unsafe_allow_html=True)
        
        video_file = open('assets/search_engine.mp4', 'rb')
        video_bytes_search_engine = video_file.read()

        st.video(video_bytes_search_engine)

        st.markdown("""

                #### 🔗 Links
                ---

                👉 The website for this project is [__available here__](https://baselhusam-search-engine-search-engine-z4stei.streamlit.app/). <br> 
                👉 The source code for this project is [__available here__](https://github.com/baselhusam/Search-Engine).

            """, unsafe_allow_html=True)

    st.write('\n')
    st.write('\n')
    st.markdown("### 🗂️ Others")
    st.markdown('---')
    st.markdown("##### 📚 The Practice of Computing Using Python Solved", unsafe_allow_html=True)
    with st.expander("The Practice of Computing Using Python Solved"):
        st.markdown(""" 
                <br>

                #### 📝 Description
                ---
                This comprehensive resource is designed to help aspiring Python learners enhance their skills and gain 
                practical experience through exercise solving. The repository includes solutions for the first 11 
                chapters of the Global Edition by Pearson.

                <br>

                #### 🔗 Links
                ---
                👉 The source code for this project is [__available here__](https://github.com/baselhusam/The-Practice-of-Computing-Using-Python-Solved).

        """, unsafe_allow_html=True)

    st.write('\n')
    st.markdown("##### 🎧 Extract Audio from YouTube Link", unsafe_allow_html=True)
    with st.expander("Extract Audio from YouTube Link"):
        st.markdown(""" 
                <br>

                #### 📝 Description
                ---
                This project is designed to extract the audio from a YouTube video link and save it as an mp3 file. 
                The project was built using the YouTube Data API and the pytube library. 

                <br>
                """, unsafe_allow_html=True)
        
        video_file = open('assets/extract_audio.mp4', 'rb')
        video_bytes_extract_audio = video_file.read()

        st.video(video_bytes_extract_audio)

        st.markdown("""
                #### 🔗 Links
                ---
                👉 The source code for this project is [__available here__](https://github.com/baselhusam/Extract-Audio-from-YouTube-Link). """, unsafe_allow_html=True)
        
# Certifications
with tabs[3]:

    st.write('\n')
    st.header("Certifications 🎖️")
    st.markdown('---')
    
    st.write('\n')
    st.markdown("##### Mathematics for Machine Learning Specialization | Coursera - Imperial College")
    with st.expander("Mathematics for Machine Learning Specialization"):

        st.write('\n')
        col1, col2, col3 = st.columns([0.25,1,0.25])
        with col2:
            img = Image.open(".//assets//certifications//Coursera//Mathematics for ML//math_ml_cert.jpg")
            st.image(img,  width=700, use_column_width='always') 

        st.write('\n')
        st.markdown(""" 

                    At the end of this specialization you will have gained the prerequisite mathematical knowledge to continue your 
                    journey and take more advanced courses in machine learning.

                    
                    1. In the first course on **Linear Algebra** we look at what linear algebra is and how it relates to data. Then we look 
                    through what vectors and matrices are and how to work with them.

                    
                    2. The second course, **Multivariate Calculus**, builds on this to look at how to optimize fitting functions to get good 
                    fits to data. It starts from introductory calculus and then uses the matrices and vectors from the first course to 
                    look at data fitting.

                    
                    3. The third course, Dimensionality Reduction with **Principal Component Analysis**, uses the mathematics from the first 
                    two courses to compress high-dimensional data. This course is of intermediate difficulty and will 
                    require Python and numpy knowledge.

                    <br>
                    
                 """, unsafe_allow_html=True)

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
    with st.expander("Machine Learning Specialization"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//Machine Learning Specialization//ml_cert.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
                    This Specialization is taught by Andrew Ng, an AI visionary who has led critical research at Stanford University and 
                    groundbreaking work at Google Brain, Baidu, and Landing.AI to advance the AI field.

                    
                    This 3-course Specialization is an updated version of Andrew’s pioneering Machine Learning course, rated 4.9 out of 5 
                    and taken by over 4.8 million learners since it launched in 2012. 

                    
                    It provides a broad introduction to modern machine learning, including supervised learning (multiple linear regression, 
                    logistic regression, neural networks, and decision trees), unsupervised learning (clustering, dimensionality reduction, 
                    recommender systems), and some of the best practices used in Silicon Valley for artificial intelligence and machine 
                    learning innovation (evaluating and tuning models, taking a data-centric approach to 
                    improving performance, and more.)


                    By the end of this Specialization, you will have mastered key concepts and gained the practical know-how to quickly 
                    and powerfully apply machine learning to challenging real-world problems. If you’re looking to break into AI 
                    or build a career in machine learning, the new Machine Learning Specialization is the best place to start.

                    <br>

        """, unsafe_allow_html=True)

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
    with st.expander("Deep Learning Specialization"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//Deep Learning Specialization//dl_cert.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 

                    In this Specialization, you will build and train neural network architectures such as Convolutional Neural Networks, 
                    Recurrent Neural Networks, LSTMs, Transformers, and learn how to make them better with strategies such as Dropout, 
                    BatchNorm, Xavier/He initialization, and more. Get ready to master theoretical concepts and their 
                    industry applications using Python and TensorFlow and tackle real-world cases such as speech 
                    recognition, music synthesis, chatbots, machine translation, natural language
                    processing, and more.
        
                    <br>
        """, unsafe_allow_html=True)

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
    with st.expander("TensorFlow Developer Certificate"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//TensorFlow Professional Certificate//tf.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        The DeepLearning.AI TensorFlow Developer Professional Certificate program teaches you applied machine learning 
        skills with TensorFlow so you can build and train powerful models. 

        In this hands-on, four-course Professional Certificate program, you’ll learn the necessary tools to build scalable 
        AI-powered applications with TensorFlow. After finishing this program, you’ll be able to apply your new TensorFlow 
        skills to a wide range of problems and projects.

        <br>

        """, unsafe_allow_html=True)

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
    with st.expander("TensorFlow: Data and Deployment Specialization"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//TensorFlow.js//tf_dep.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        In this four-course Specialization, you’ll learn how to get your machine learning models into the hands 
        of real people on all kinds of devices. Start by understanding how to train and run machine learning models in browsers
        and in mobile applications. Learn how to leverage built-in datasets with just a few lines of code, learn about data pipelines
        with TensorFlow data services, use APIs to control data splitting, process all types of unstructured data, and retrain deployed 
        models with user data while maintaining data privacy.  Apply your knowledge in various deployment scenarios and get introduced to 
        TensorFlow Serving, TensorFlow, Hub, TensorBoard, and more. 

        <br>
        """, unsafe_allow_html=True)

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
    with st.expander("Natural Language Processing Specialization"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//NLP Specialization//nlp.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        By the end of this Specialization, you will be ready to design NLP applications that perform question-answering and 
        sentiment analysis, create tools to translate languages and summarize text, and even build chatbots. These and other 
        NLP applications are going to be at the forefront of the coming transformation to an AI-powered future.

        <br>
            
        """, unsafe_allow_html=True)

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
    st.markdown("##### AI For Medicine Specialization | Coursera - DeepLearning.AI")
    with st.expander("AI For Medicine Specialization"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//AI For Medicine//ai_med.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        AI is transforming the practice of medicine. It’s helping doctors diagnose patients more accurately, make predictions 
        about patients’ future health, and recommend better treatments. This three-course Specialization will give you practical 
        experience in applying machine learning to concrete problems in medicine.


        <br>

        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h5 align='center'> AI for Medical Diagnosis </h6>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//AI For Medicine//ai_med1.jpg")
            st.image(img,  width=200, use_column_width='always')


        with col2:
            st.markdown("<h5 align='center'> AI for Medical Treatment </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//AI For Medicine//ai_med2.jpg")
            st.image(img,  width=200, use_column_width='always')

        with col3:
            st.markdown("<h5 align='center'> AI for Medical Prognosis </h5>", unsafe_allow_html=True)
            img = Image.open(".//assets//certifications//Coursera//AI For Medicine//ai_med3.jpg")
            st.image(img,  width=200, use_column_width='always')

    st.write('\n')
    st.markdown("##### AWS Fundementals Specialization | Coursera - AWS")
    with st.expander("AWS Fundementals Specialization"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//AWS Specialization//aws.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        This specialization gives current or aspiring IT professionals an overview of the features, benefits, and capabilities 
        of Amazon Web Services (AWS). As you proceed through these four interconnected courses, you will gain a more vivid 
        understanding of core AWS services, key AWS security concepts, strategies for migrating from on-premises to AWS, 
        and basics of building serverless applications with AWS. Additionally, you will have opportunities to practice 
        what you have learned by completing labs and exercises developed by AWS technical instructors.

        <br>

        """, unsafe_allow_html=True)

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
    with st.expander("Data Visualization with Python"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//data_vis.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        This course will teach you to work with many Data Visualization tools and techniques. You will learn to create various 
        types of basic and advanced graphs and charts like: Waffle Charts, Area Plots, Histograms, Bar Charts, Pie Charts, 
        Scatter Plots, Word Clouds, Choropleth Maps, and many more! You will also create interactive dashboards that 
        allow even those without any Data Science experience to better understand data, and make more 
        effective and informed decisions.   

        You will learn hands-on by completing numerous labs and a final project to practice and apply the many aspects and 
        techniques of Data Visualization using Jupyter Notebooks and a Cloud-based IDE. You will use several data visualization 
        libraries in Python, including Matplotlib, Seaborn, Folium, Plotly & Dash.

        <br>

        """, unsafe_allow_html=True)

    st.write('\n')
    st.markdown("##### Machine Learning with Python | Coursera - IBM")
    with st.expander("Machine Learning with Python"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//ml_python.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        This course will begin with a gentle introduction to Machine Learning and what it is, with topics like supervised vs 
        unsupervised learning, linear & non-linear regression, simple regression and more.  

        You will then dive into classification techniques using different classification algorithms, namely K-Nearest Neighbors (KNN), 
        decision trees, and Logistic Regression. You’ll also learn about the importance and different types of clustering such as k-means, 
        hierarchical clustering, and DBSCAN. 
        
        <br>

        """, unsafe_allow_html=True)

    st.write('\n')
    st.markdown("##### Python Project for Data Engineering | Coursera - IBM")
    with st.expander("Python Project for Data Engineering"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Coursera//data_eng_proj.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        By the end of this project, you will have demonstrated your familiarity with important skills in Information Engineering and Extraction, 
        Transformation and Loading (ETL), Jupyter Notebooks, and of course, Python Programming. 

        Upon completion of this course, you will have acquired the confidence to begin collecting large datasets, webscraping, using APIs, and 
        performing ETL tasks, to hone valuable data management skills - all with the use of Python.  

        <br>

        """, unsafe_allow_html=True)

    st.write('\n')
    st.markdown("##### Data Cleaning & Preprocessing with Pandas | 365 days of Data Science")
    with st.expander("Data Cleaning & Preprocessing with Pandas"):
        st.write('\n')

        col1, col2, col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//365 Data Science//data_clean.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        This course introduces you to one of the most widely used data analysis libraries – pandas. With pandas you’ll be able to solve your 
        analytic tasks in an easy and professional way!

        1. Develop a basic understanding of the pandas library 
        2. Navigate through the pandas documentation 
        3. Practice with fundamental programming tools 
        4. Study collecting, cleaning, and preprocessing data 
        5. Work with pandas Series and DataFrames 
        6. Practice data selection with pandas 

        <br>

        """, unsafe_allow_html=True)

    st.write('\n')
    st.markdown("##### Macine Learning Intern | SteamCenter")
    with st.expander("Machine Learning Intern | SteamCenter"):
        st.write('\n')

        col1, col2,col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//steam_center.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
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

    st.write('\n')
    st.markdown("##### AWS Machine Learning Foundation | Udacity - AWS")
    with st.expander("AWS Machine Learning Foundation"): 
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
        st.markdown(""" 
        
        Learn what machine learning is and the steps involved in building and evaluating models. Gain in demand skills needed at 
        businesses working to solve challenges with AI.

        Learn the fundamentals of advanced machine learning areas such as computer vision, reinforcement learning, and generative AI. 
        Get hands-on with machine learning using AWS AI Devices (i.e. AWS DeepRacer and AWS DeepComposer).

        <br>

        """, unsafe_allow_html=True)

    st.write('\n')
    st.markdown("##### Data Analysis Track | Udacity")
    with st.expander("Data Analysis Track"):
        st.write('\n')

        col1, col2,col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Udacity//DataAnalysis__Udacity.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        Advance your programming skills and refine your ability to work with messy, complex datasets. You’ll learn 
        to manipulate and prepare data for analysis, and create visualizations for data exploration. Finally, you’ll 
        learn to use your data skills to tell a story with data.

        <br>

        """, unsafe_allow_html=True)

    st.write('\n')
    st.markdown("##### Data Engineering For Everyone | Udacity")
    with st.expander("Data Engineering For Everyone"):
        st.write('\n')

        col1, col2,col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Data camp//data_eng.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        In this course, you’ll learn about a data engineer’s core responsibilities, how they differ from data 
        scientists, and facilitate the flow of data through an organization. Through hands-on exercises you’ll 
        follow Spotflix, a fictional music streaming company, to understand how their data engineers collect, 
        clean, and catalog their data. By the end of the course, you’ll understand what your company's data 
        engineers do, be ready to have a conversation with a data engineer, and have a solid foundation to 
        start your own data engineer journey.

        <br>

        """, unsafe_allow_html=True)

    st.write('\n')
    st.markdown("##### Introduction to Python | DataCamp")
    with st.expander("Introduction to Python"):
        st.write('\n')

        col1, col2,col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Data camp//python.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        Python has grown to become the market leader in programming languages and the language of choice for data analysts and data 
        scientists. Demand for data skills is rising because companies want to gain actionable insights from their data.

        <br>

        ###### Discover the Python Basics
        This is a Python course for beginners, and we designed it for people with no prior Python experience. It is even 
        suitable if you have no coding experience at all. You will cover the basics of Python, helping you understand common, 
        everyday functions and applications, including how to use Python as a calculator, understanding variables and types, 
        and building Python lists. The first half of this course prepares you to use Python interactively and teaches you how to store, 
        access, and manipulate data using one of the most popular programming languages in the world.
        
        <br>

        ###### Explore Python Functions and Packages
        The second half of the course starts with a view of how you can use functions, methods, and packages to use code that other 
        Python developers have written. As an open-source language, Python has plenty of existing packages and libraries that you can 
        use to solve your problems.

        <br>

        ###### Get Started with NumPy
        NumPy is an essential Python package for data science. You’ll finish this course by learning to use some of the most popular 
        tools in the NumPy array and start exploring data in Python.

        <br>

        """, unsafe_allow_html=True)

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
        st.markdown(""" 
        
        These are 6 LinkedIn Certificates that I have earned. All of them are talking about Data Analysis and Data Science. 
        Which is the main reason I am here.

        <br>

        """, unsafe_allow_html=True)

    st.write('\n')
    st.markdown("##### IEEE Extreme | IEEE")
    with st.expander("IEEE Extreme"):
        st.write('\n')

        col1, col2,col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//IEEE//IEEE Extreme.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        This is a certificate of participation in IEEE Extreme. IEEE Extreme is a 24-hour programming competition.
        It is a great opportunity to test your programming skills and learn new things.

        <br>

        """, unsafe_allow_html=True)

    st.write('\n')
    st.markdown("##### Python Programming | Udemy")
    with st.expander("Python Programming | Udemy"):
        st.write('\n')

        col1, col2,col3 = st.columns([0.25,1,0.25])

        with col2:
            img = Image.open(".//assets//certifications//Udemy//python.jpg")
            st.image(img,  width=700, use_column_width='always')

        st.write('\n')
        st.markdown(""" 
        
        This is a certificate of completion in Python Programming. This course is a great introduction to Python.
        It is a great course for beginners.

        <br>

        """, unsafe_allow_html=True)
  
# Volunteering
with tabs[4]:

    st.write('\n')
    st.header("Volunteering 🤝")
    st.markdown('---')

    st.write('\n')
    st.markdown("""#### Technical Team Lead at IEEE CIS""", unsafe_allow_html=True)
    with st.expander("IEEE CIS"):

        col1, col2, col3 = st.columns([0.25,1,0.25])
        with col2:
            st.write('\n')
            img = Image.open(".//assets//ieee_cis_logo.png")
            st.image(img,  width=700, use_column_width='always')


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

                    The GitHub repository of this workshop is [__here__](https://github.com/baselhusam/Titanic-Dataset---Data-Preparation).
                    <br> <br>

                    #### Data Preparation Course

                    In this course I've talked about the importance of data preparation, and I've went deep into the data preparation phases,
                    from the data collection phase to the splitting the data into training and testing sets. This course is a 15 hours course, with 
                    a final project that is related to data preparation, each student has to prepare a dataset and apply the different steps of data preparation
                    on his own project.

                    The GitHub repository of this course is [__here__](https://github.com/baselhusam/Data-Preparation-Course-IEEE-CIS).
                    """, unsafe_allow_html=True)
    
    st.write('\n')
    st.markdown(""" #### Director of the ACM JU Magazine & Contributing Writer & Audio Eng""", unsafe_allow_html=True)
    with st.expander("ACM JU Magazine"):
        
        col1, col2, col3 = st.columns([0.25,1,0.25])
        with col2:
            st.write('\n')
            img = Image.open(".//assets//acm_logo.png")
            st.image(img,  width=600, use_column_width='always')

        st.write('\n')
        st.markdown("""
        
                    
                    ##### Director of the ACM JU Magazine

                    As a Director of the ACM JU Magazine, my role is to manage the magazine, and the magazine contains 
                    articles, interviews, and news about the ACM JU Chapter.

                    I've been the Director of this Magazine since the 13th issue.
                    
                    <br> 

                    ##### Contributing Writer

                    As a Contributing Writer, I've written 6 articles that are related to Data Science, Machine Learning, and AI.

                    1. What is Big Data? From the 9th issue of the Magazine. [__Link__](https://issuu.com/acm_ju/docs/acm_ju_magazine_9th_edition)
                    2. Data Science in Astronomy. From the 10th issue of the Magazine. [__Link__](https://issuu.com/acm_ju/docs/acm_ju_magazine_10th_edition)
                    3. A Deeper Look into Machine Learning. From the 11th issue of the Magazine. [__Link__](https://issuu.com/acm_ju/docs/acm_ju_magazine_11th_edition)
                    4. The Future with Time Series Forecasting. From the 12th issue of the Magazine. [__Link__](https://issuu.com/acm_ju/docs/acm_ju_magazine_12th_edition)
                    5. Can Computers See? From the 13th issue of the Magazine. [__Link__](https://issuu.com/acm_ju/docs/acm_ju_magazine_13th_edition)
                    6. Make Your Story Go Viral. From the 14th issue of the Magazine. [__Link__](https://issuu.com/acm_ju/docs/acm_ju_magazine_14th_edition)

                    <br>

                    ##### Audio Engineer & Music Producer
                    
                    We wanted to have a podcast for the Magazine, so we've started to record the podcast, make the audio engineering & music production for 
                    the podcast, to be ready to be published on the listening platforms, and to be proffesional as much as we can. 

                    As an Audio Engineer & Music Producer, my role is to make audio engiineering for the vocals by the podcasters of the Magazine, so
                    we can have a good quality podcast and put it on the listenings platforms. Also, I've produced the music for the podcast intro, outro, 
                    and the music between articles.

                    You can listen to the podcast on Spotify [__here__](https://open.spotify.com/show/3Bnw1rRtPJ1Pv3pC5GOr9g?si=3185c3b28f284590).


                    """, unsafe_allow_html=True)
        
    st.write('\n')
    st.markdown(""" #### Member at ITeam JU """, unsafe_allow_html=True)
    with st.expander("ITeam JU"):
        
        col1, col2, col3 = st.columns([0.25,1,0.25])
        with col2:
            st.write('\n')
            st.write('\n')
            img = Image.open(".//assets//iteam_logo.png")
            st.image(img,  width=600, use_column_width='always')

        st.write('\n')
        st.markdown("""

                    ITeam JU as a collections of students who are interested in the field of Information Technology.

                    As a Member at ITeam JU, I was at the Achademic Team, and my role was to help the students with their subject,
                    I made a Summary for a subject at the university called Data Engineering and Analytics. 
                    You can find the summary from here [__Link__](https://drive.google.com/file/d/1kVg2KB4VHcrBlgJ6HE9HJ5CALfJos4_O/view?mcp_token=eyJwaWQiOjEwMjgyNzE3NTQwNzQxMiwic2lkIjo2MTk5MzU0NzQzNDE5MTY1LCJheCI6ImM1MTE0ZGM5YzA3YTQ4ODA4NjAxMmMyMDI1NTZkZGM2IiwidHMiOjE2ODE1NzM1ODAsImV4cCI6MTY4Mzk5Mjc4MH0.1lxNFMxFdz0ObNUkft5dV6ABFn-flyesnyp3K0Ni9ps).

                    <br>

                    Also, I made 3 Online Courses that are related to Data Science, Machine Learning, and AI.
                    1. Principles of Data Sciense. [__Link__](https://www.youtube.com/playlist?list=PLd2pEan0ZG_b-fat4bMOLdiyS-rk52r3T).
                    2. Data Engineering & Analytics. [__Link__](https://www.youtube.com/playlist?list=PLd2pEan0ZG_bkO6R4qdUyyzjcCj5n6LJt).
                    3. Cloud Computing Project. [__Link__](https://www.youtube.com/playlist?list=PLd2pEan0ZG_beugQoHfBmGwyLzvSoS6Q-).
                    4. Machine Learning & Neural Networks. [__Link__](https://www.youtube.com/playlist?list=PLd2pEan0ZG_bZS9l0RkJ8D3xqn74z1kor).
        
                    <br> 

                    """, unsafe_allow_html=True)
    
    st.write('\n')
    st.markdown(""" #### Ambassador at SHAI FOR AI""", unsafe_allow_html=True)
    with st.expander("SHAI FOR AI"):
        st.write('\n')

        st.markdown("""
        
                    As an Ambassador at SHAI FOR AI, my role is to promote the SHAI FOR AI community, and the community contains 
                    courses, workshops, and competitions.

                    Also, I've been event organizer for the SHAI FOR AI community, and I've organized 2 events that are related 
                    to Data Science, Machine Learning, and AI. One of the events was a Bootcamp that was about a Kaggle competition,
                    and this bootcamp was a 12 hours bootcamp, and we've started from zero to hero, and we've talked about the
                    different steps of a machine learning project, and Tips & Tricks for the ML competitions so we can win the competition,
                    and avoid the common mistakes that we can make in the competition.

                    I've been an Ambassador at SHAI FOR AI since Nov 2022 - Jan 2023.

                    <br> 

                    """, unsafe_allow_html=True)

# Recommendation Letter
with tabs[5]:
    
    st.write("\n")
    st.title("Recommendation Letters ✉️")
    st.markdown("---")
    st.write("""
    This section contains recommendation letters that provide more information about my professional background and skills.
    """)

    st.write("\n")

    st.subheader("Prof. Ibrahim Aljarah")
    st.markdown("""Contact information: [__Email__](mailto:i.aljarah@ju.edu.jo)
                """)
    
    col1, col2, col3 = st.columns([0.1,1,0.1])
    col2.image(".//assets//Ibrahim-Rec-Let.jpg")

    st.write("\n\n")

    st.subheader("Dr. Sherenaz Al-Haj Baddar")
    st.markdown("""
                Contact information: [__Email__](mailto:s.baddar@ju.edu.jo) \n
                **Telephone:** +962 06 535000 ext. 22589
                """)
    col1, col2, col3 = st.columns([0.1,1,0.1])
    col2.image(".//assets//Sherenaz-Rec-Let.jpg")
