import streamlit as st
import mysql.connector
import pandas as pd
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import bcrypt
import random

# Connect to the database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="skill_navigator"
)
cursor = db.cursor(buffered=True)

# Streamlit app configuration
st.set_page_config(page_title="Skill Navigator", page_icon=":brain:")

def initialize_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None

def hash_password(password):
    """Hash a password for storing."""
    # Convert the password to bytes
    password_bytes = password.encode('utf-8')
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    # Return the hashed password as a string
    return hashed.decode('utf-8')

def check_password(password, hashed_password):
    """Check a password against a stored hash."""
    try:
        # Convert both strings to bytes for comparison
        password_bytes = password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        st.error(f"Password verification error: {str(e)}")
        return False
    
def candidate_login():
    st.title("Candidate Login")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if not email or not password:
                st.error("Please enter both email and password")
                return

            try:
                # Fetch user data
                cursor.execute("""
                    SELECT candidate_id, name, email, hashed_password, degree, specialization, 
                           phone_number, linkedin_profile, github_profile 
                    FROM candidates 
                    WHERE email = %s
                """, (email,))
                result = cursor.fetchone()

                if result:
                    stored_hash = result[3]  # Get stored hash
                    if check_password(password, stored_hash):
                        st.session_state.logged_in = True
                        st.session_state.user_data = result
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid password")
                else:
                    st.error("Email not found")
            except mysql.connector.Error as err:
                st.error(f"Database error: {err}")

def candidate_dashboard():
    if not st.session_state.logged_in:
        st.warning("Please login first")
        return
        
    candidate_data = st.session_state.user_data
    st.title(f"Welcome, {candidate_data[1]}!")

    # Display candidate information
    st.subheader("Personal Information")
    st.write(f"Email: {candidate_data[2]}")
    st.write(f"Degree: {candidate_data[4]}")
    st.write(f"Specialization: {candidate_data[5]}")
    st.write(f"Phone Number: {candidate_data[6]}")

    # Display social media profiles
    st.subheader("Social Media")
    st.write(f"LinkedIn: {candidate_data[7]}")
    st.write(f"GitHub: {candidate_data[8]}")

    # Display training progress
    st.subheader("Training Progress")
    
    # Course progress data (same as in display_training_progress function)
    courses = {
        "HTML": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-1.png"},
        "CSS": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-2.png"},
        "JavaScript": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-3.png"},
        "Bootstrap": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-4.png"},
        "jQuery": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-5.png"},
        "SASS": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-6.png"},
        "PHP": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-7.png"},
        "MySQL": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-8.png"},
        "React": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-9.png"}
    }
    
    progress = {
        "HTML": 8,
        "CSS": 6,
        "JavaScript": 4,
        "Bootstrap": 0,
        "jQuery": 0,
        "SASS": 0,
        "PHP": 0,
        "MySQL": 0,
        "React": 0
    }
    
    # Calculate overall progress
    total_videos_completed = sum(progress.values())
    total_videos = sum(course["videos"] for course in courses.values())
    overall_progress = (total_videos_completed / total_videos) * 100
    
    # Display overall progress
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Overall Progress", f"{overall_progress:.1f}%")
        st.progress(overall_progress / 100)
    
    with col2:
        st.metric("Completed Courses", f"{total_videos_completed}/{total_videos} videos")
    
    # Display individual course progress
    progress_data = pd.DataFrame({
        'Course': list(progress.keys()),
        'Completed': list(progress.values()),
        'Total': [courses[course]["videos"] for course in progress.keys()]
    })
    
    progress_data['Percentage'] = (progress_data['Completed'] / progress_data['Total'] * 100).round(1)
    
    # Create a more compact visualization for the dashboard
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=progress_data['Course'],
        y=progress_data['Percentage'],
        marker_color='rgb(158,202,225)',
        text=progress_data['Percentage'].apply(lambda x: f'{x:.1f}%'),
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Course-wise Progress',
        xaxis_title='Course',
        yaxis_title='Completion %',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Display feedback section
    st.subheader("Feedback")
    st.info("Feedback system coming soon!")

def logout():
    st.session_state.logged_in = False
    st.session_state.user_data = None
    st.rerun()

def get_course_progress(candidate_id):
    """Fetch course progress for a candidate"""
    cursor.execute("""
        SELECT cp.course_id, c.course_name, cp.videos_completed, c.total_videos, 
               cp.completion_date, cp.status
        FROM course_progress cp
        JOIN courses c ON cp.course_id = c.course_id
        WHERE cp.candidate_id = %s
    """, (candidate_id,))
    return cursor.fetchall()

def get_course_videos():
    """Get video URLs for each course"""
    # Example YouTube video IDs for each course
    # In production, these would come from your database
    course_videos = {
        "HTML": [
            {"title": "HTML Basics - Introduction", "url": "https://www.youtube.com/watch?v=qz0aGYrrlhU"},
            {"title": "HTML Elements and Tags", "url": "https://www.youtube.com/watch?v=UB1O30fR-EE"},
            {"title": "HTML Forms and Input", "url": "https://www.youtube.com/watch?v=fNcJuPIZ2WE"},
            {"title": "HTML Tables and Lists", "url": "https://www.youtube.com/watch?v=N69xumSjg5Q"},
            {"title": "HTML Semantic Elements", "url": "https://www.youtube.com/watch?v=kGW8Al_cga4"},
            {"title": "HTML Media Elements", "url": "https://www.youtube.com/watch?v=B5EH6dQ9_Yo"},
            {"title": "HTML Layout Elements", "url": "https://www.youtube.com/watch?v=PlxWf493en4"},
            {"title": "HTML Best Practices", "url": "https://www.youtube.com/watch?v=bWPMSSsVdPk"},
            {"title": "HTML5 Features", "url": "https://www.youtube.com/watch?v=DPnqb74Smug"},
            {"title": "HTML Project", "url": "https://www.youtube.com/watch?v=hdI2bqOjy3c"}
        ],
        "CSS": [
            {"title": "CSS Basics", "url": "https://www.youtube.com/watch?v=1PnVor36_40"},
            # Add more videos for CSS
        ],
        # Add similar video lists for other courses
    }
    return course_videos

def display_training_progress():
    if not st.session_state.logged_in:
        st.warning("Please login first")
        return
        
    candidate_data = st.session_state.user_data
    st.title("Training Progress")
    
    # Available courses data
    courses = {
        "HTML": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-1.png"},
        "CSS": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-2.png"},
        "JavaScript": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-3.png"},
        "Bootstrap": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-4.png"},
        "jQuery": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-5.png"},
        "SASS": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-6.png"},
        "PHP": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-7.png"},
        "MySQL": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-8.png"},
        "React": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-9.png"}
    }
    
    course_videos = get_course_videos()
    
    # Course Selection
    st.subheader("Available Courses")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_course = st.selectbox(
            "Select a course to view or start:",
            list(courses.keys())
        )
    
    with col2:
        if st.button("Enroll in Course"):
            # Add enrollment logic here
            st.success(f"Enrolled in {selected_course} course!")
            
    # Display selected course details
    st.subheader(f"{selected_course} Course Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Instructor", courses[selected_course]["tutor"])
    with col2:
        st.metric("Total Videos", courses[selected_course]["videos"])
    with col3:
        st.metric("Course Status", "In Progress")
        
    # Progress tracking
    st.subheader("Your Progress")
    progress = {
        "HTML": 8,
        "CSS": 6,
        "JavaScript": 4,
        "Bootstrap": 0,
        "jQuery": 0,
        "SASS": 0,
        "PHP": 0,
        "MySQL": 0,
        "React": 0
    }
    
    # Progress bar for selected course
    videos_completed = progress[selected_course]
    total_videos = courses[selected_course]["videos"]
    progress_percentage = videos_completed / total_videos
    display_percentage = progress_percentage * 100
    
    st.progress(progress_percentage)
    st.write(f"Completed {videos_completed} out of {total_videos} videos ({display_percentage:.1f}%)")
    
    # Overall progress visualization
    st.subheader("Overall Training Progress")
    
    progress_data = pd.DataFrame({
        'Course': list(progress.keys()),
        'Completed': list(progress.values()),
        'Total': [courses[course]["videos"] for course in progress.keys()]
    })
    
    progress_data['Percentage'] = (progress_data['Completed'] / progress_data['Total'] * 100).round(1)
    
    fig = px.bar(progress_data,
                 x='Course',
                 y='Percentage',
                 title='Course Completion Status',
                 labels={'Percentage': 'Completion %'},
                 color='Percentage',
                 color_continuous_scale='blues')
    
    st.plotly_chart(fig)
    
    # Recent activity
    st.subheader("Recent Activity")
    activity_data = [
        {"date": "2024-11-09", "activity": "Completed HTML Module 8: Forms"},
        {"date": "2024-11-08", "activity": "Started CSS Module 7: Flexbox"},
        {"date": "2024-11-07", "activity": "Completed JavaScript Module 4: Functions"}
    ]
    
    for activity in activity_data:
        st.info(f"{activity['date']}: {activity['activity']}")
        
    # Add video playlist for selected course
    st.subheader("Course Videos")
    
    # Check if videos exist for the selected course
    if selected_course in course_videos:
        videos = course_videos[selected_course]
        
        # Create tabs for video list and video player
        tab1, tab2 = st.tabs(["Video List", "Video Player"])
        
        with tab1:
            for i, video in enumerate(videos):
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"📺 {video['title']}", key=f"video_button_{i}"):
                        st.session_state.current_video = video['url']
                        st.session_state.current_video_title = video['title']
                with col2:
                    if i < progress[selected_course]:
                        st.success("Completed ✓")
                    else:
                        st.button("Mark Complete", key=f"complete_{i}")
        
        with tab2:
            if 'current_video' in st.session_state:
                st.subheader(st.session_state.current_video_title)
                # Extract video ID from YouTube URL
                video_id = st.session_state.current_video.split('watch?v=')[-1]
                # Embed YouTube video
                st.markdown(f'''
                    <iframe width="100%" height="400" src="https://www.youtube.com/embed/{video_id}" 
                    frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; 
                    gyroscope; picture-in-picture" allowfullscreen></iframe>
                    ''', unsafe_allow_html=True)
            else:
                st.info("Select a video from the Video List tab to start watching")
    else:
        st.info(f"Videos for {selected_course} course will be available soon!")
def display_feedback():
    if not st.session_state.logged_in:
        st.warning("Please login first")
        return
        
    candidate_data = st.session_state.user_data
    st.title("Training Feedback")
    
    # Get course progress data
    progress = {
        "HTML": 8,
        "CSS": 6,
        "JavaScript": 4,
        "Bootstrap": 0,
        "jQuery": 0,
        "SASS": 0,
        "PHP": 0,
        "MySQL": 0,
        "React": 0
    }
    
    # Topic completion dates (this would come from database in production)
    completed_topics = [
        {"course": "HTML", "topic": "HTML Basics", "completion_date": "2024-11-08"},
        {"course": "HTML", "topic": "HTML Forms", "completion_date": "2024-11-09"},
        {"course": "CSS", "topic": "CSS Basics", "completion_date": "2024-11-07"},
    ]
    
    # Display feedback sections
    st.subheader("Submit Topic Feedback")
    
    # Topic selection
    courses_with_progress = [course for course, completed in progress.items() if completed > 0]
    selected_course = st.selectbox("Select Course", courses_with_progress)
    
    # Get topics for selected course
    topics = {
        "HTML": ["HTML Basics", "HTML Elements", "HTML Forms", "HTML Tables", "HTML Semantic Elements",
                "HTML Media", "HTML Layout", "HTML Best Practices"],
        "CSS": ["CSS Basics", "CSS Selectors", "CSS Box Model", "CSS Flexbox", "CSS Grid", "CSS Animations"],
        "JavaScript": ["JS Basics", "JS Functions", "JS DOM", "JS Events"]
    }
    
    if selected_course in topics:
        selected_topic = st.selectbox("Select Topic", topics[selected_course])
        
        with st.form("feedback_form"):
            st.write("Please rate the following aspects (1-5):")
            content_rating = st.slider("Content Quality", 1, 5, 3)
            clarity_rating = st.slider("Clarity of Explanation", 1, 5, 3)
            practical_rating = st.slider("Practical Relevance", 1, 5, 3)
            pace_rating = st.slider("Learning Pace", 1, 5, 3)
            
            st.write("Additional Feedback:")
            understanding = st.select_slider(
                "How well did you understand the topic?",
                options=["Not at all", "Slightly", "Moderately", "Very well", "Completely"],
                value="Moderately"
            )
            
            difficulty = st.select_slider(
                "How difficult was the topic?",
                options=["Very Easy", "Easy", "Moderate", "Difficult", "Very Difficult"],
                value="Moderate"
            )
            
            challenges = st.text_area("What challenges did you face while learning this topic?")
            suggestions = st.text_area("Do you have any suggestions for improvement?")
            
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                # Here you would save the feedback to database
                st.success("Thank you for your feedback!")
                
    # Display feedback history
    st.subheader("Your Feedback History")
    
    # Sample feedback history (would come from database in production)
    feedback_history = [
        {
            "date": "2024-11-09",
            "course": "HTML",
            "topic": "HTML Forms",
            "ratings": {"content": 4, "clarity": 5, "practical": 4, "pace": 3},
            "understanding": "Very well",
            "difficulty": "Moderate"
        },
        {
            "date": "2024-11-08",
            "course": "HTML",
            "topic": "HTML Basics",
            "ratings": {"content": 5, "clarity": 4, "practical": 5, "pace": 4},
            "understanding": "Completely",
            "difficulty": "Easy"
        }
    ]
    
    for feedback in feedback_history:
        with st.expander(f"{feedback['course']} - {feedback['topic']} ({feedback['date']})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Ratings:")
                for aspect, rating in feedback['ratings'].items():
                    st.write(f"- {aspect.title()}: {rating}/5")
            
            with col2:
                st.write(f"Understanding: {feedback['understanding']}")
                st.write(f"Difficulty Level: {feedback['difficulty']}")
    
    # Display feedback analysis
    st.subheader("Feedback Analysis")
    
    # Calculate average ratings
    if feedback_history:
        avg_ratings = {
            "Content": sum(f['ratings']['content'] for f in feedback_history) / len(feedback_history),
            "Clarity": sum(f['ratings']['clarity'] for f in feedback_history) / len(feedback_history),
            "Practical": sum(f['ratings']['practical'] for f in feedback_history) / len(feedback_history),
            "Pace": sum(f['ratings']['pace'] for f in feedback_history) / len(feedback_history)
        }
        
        # Create visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(avg_ratings.keys()),
            y=list(avg_ratings.values()),
            marker_color='rgb(158,202,225)',
            text=[f"{v:.1f}/5" for v in avg_ratings.values()],
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Your Average Ratings by Category',
            yaxis_title='Rating (out of 5)',
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(range=[0, 5])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # AI-generated recommendations based on feedback
        st.subheader("AI Recommendations")
        
        # Sample recommendations (in production, these would come from AI analysis)
        avg_understanding = sum(1 for f in feedback_history if f['understanding'] in ['Very well', 'Completely']) / len(feedback_history)
        
        if avg_understanding < 0.7:
            st.warning("Consider reviewing previous topics before moving forward. Your understanding scores indicate some gaps.")
        else:
            st.success("You're making good progress! Consider taking on more challenging exercises.")
            
        # Topic-specific recommendations
        st.write("Topic-specific recommendations:")
        recommendations = {
            "HTML": "Focus on practicing more with forms and tables",
            "CSS": "Try building more complex layouts using Flexbox",
            "JavaScript": "Work on additional DOM manipulation exercises"
        }
        
        for course, recommendation in recommendations.items():
            if any(f['course'] == course for f in feedback_history):
                st.info(f"{course}: {recommendation}")

# MCQ questions database (In production, this would be in the database)
MCQ_QUESTIONS = {
    "HTML": [
        {
            "question": "What does HTML stand for?",
            "options": [
                "Hyper Text Markup Language",
                "High Text Markup Language",
                "Hyper Tabular Markup Language",
                "None of these"
            ],
            "correct_answer": 0
        },
        {
            "question": "Which tag is used to create a hyperlink?",
            "options": ["<link>", "<a>", "<href>", "<url>"],
            "correct_answer": 1
        },
        # Add more questions for HTML
    ],
    "CSS": [
        {
            "question": "What does CSS stand for?",
            "options": [
                "Cascading Style Sheets",
                "Computer Style Sheets",
                "Creative Style Sheets",
                "Colorful Style Sheets"
            ],
            "correct_answer": 0
        },
        # Add more questions for CSS
    ],
    "JavaScript": [
        {
            "question": "Which keyword is used to declare variables in JavaScript?",
            "options": ["var", "let", "const", "All of the above"],
            "correct_answer": 3
        },
        # Add more questions for JavaScript
    ]
}

def display_mcq_test():
    if not st.session_state.logged_in:
        st.warning("Please login first")
        return
        
    st.title("MCQ Tests")
    
    # Get available courses
    available_courses = list(MCQ_QUESTIONS.keys())
    
    # Test selection
    selected_course = st.selectbox("Select Course for Test", available_courses)
    
    if selected_course:
        st.subheader(f"MCQ Test: {selected_course}")
        
        # Check if there's an ongoing test
        if 'current_test' not in st.session_state:
            if st.button("Start Test"):
                # Initialize test
                st.session_state.current_test = {
                    'course': selected_course,
                    'questions': random.sample(MCQ_QUESTIONS[selected_course], min(10, len(MCQ_QUESTIONS[selected_course]))),
                    'answers': [],
                    'score': 0,
                    'start_time': datetime.now()
                }
                st.rerun()
        
        # Display ongoing test
        if 'current_test' in st.session_state:
            with st.form("mcq_test"):
                for i, question in enumerate(st.session_state.current_test['questions']):
                    st.write(f"\nQ{i+1}. {question['question']}")
                    answer = st.radio(
                        f"Select your answer for question {i+1}:",
                        question['options'],
                        key=f"q_{i}"
                    )
                    st.session_state.current_test['answers'].append(
                        question['options'].index(answer)
                    )
                
                submitted = st.form_submit_button("Submit Test")
                if submitted:
                    # Calculate score
                    correct_answers = 0
                    for i, question in enumerate(st.session_state.current_test['questions']):
                        if st.session_state.current_test['answers'][i] == question['correct_answer']:
                            correct_answers += 1
                    
                    # Calculate percentage
                    score_percentage = (correct_answers / len(st.session_state.current_test['questions'])) * 100
                    
                    # Store test results
                    test_result = {
                        'course': selected_course,
                        'score': score_percentage,
                        'date': st.session_state.current_test['start_time'],
                        'duration': (datetime.now() - st.session_state.current_test['start_time']).seconds,
                        'correct_answers': correct_answers,
                        'total_questions': len(st.session_state.current_test['questions'])
                    }
                    
                    if 'test_history' not in st.session_state:
                        st.session_state.test_history = []
                    
                    st.session_state.test_history.append(test_result)
                    
                    # Clear current test
                    del st.session_state.current_test
                    
                    # Show results
                    st.success(f"Test completed! Your score: {score_percentage:.1f}%")
                    st.rerun()

def display_test_reports():
    if not st.session_state.logged_in:
        st.warning("Please login first")
        return
        
    st.title("Test Reports")
    
    if 'test_history' not in st.session_state or not st.session_state.test_history:
        st.info("No test history available. Complete some tests to see your reports.")
        return
    
    # Overall Performance
    st.subheader("Overall Performance")
    
    # Calculate statistics
    total_tests = len(st.session_state.test_history)
    avg_score = sum(test['score'] for test in st.session_state.test_history) / total_tests
    best_score = max(test['score'] for test in st.session_state.test_history)
    worst_score = min(test['score'] for test in st.session_state.test_history)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Score", f"{avg_score:.1f}%")
    with col2:
        st.metric("Best Score", f"{best_score:.1f}%")
    with col3:
        st.metric("Tests Taken", total_tests)
    
    # Performance Trend
    st.subheader("Performance Trend")
    
    # Prepare data for visualization
    trend_data = pd.DataFrame(st.session_state.test_history)
    trend_data['date'] = pd.to_datetime(trend_data['date'])
    trend_data = trend_data.sort_values('date')
    
    fig = px.line(trend_data, 
                  x='date', 
                  y='score',
                  title='Test Score Trend',
                  labels={'date': 'Test Date', 'score': 'Score (%)'},
                  markers=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Course-wise Performance
    st.subheader("Course-wise Performance")
    
    course_data = trend_data.groupby('course').agg({
        'score': ['mean', 'min', 'max', 'count']
    }).round(1)
    
    course_data.columns = ['Average Score', 'Lowest Score', 'Highest Score', 'Tests Taken']
    st.dataframe(course_data)
    
    # Detailed Test History
    st.subheader("Test History")
    
    for test in reversed(st.session_state.test_history):
        with st.expander(f"{test['course']} - {test['date'].strftime('%Y-%m-%d %H:%M')}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Score: {test['score']:.1f}%")
                st.write(f"Correct Answers: {test['correct_answers']}/{test['total_questions']}")
            with col2:
                st.write(f"Duration: {test['duration']} seconds")
                status = "Pass" if test['score'] >= 70 else "Fail"
                if status == "Pass":
                    st.success(status)
                else:
                    st.error(status)
    
    # Export Report
    if st.button("Export Report"):
        # Create Excel-like report string
        report = "Test Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        report += f"Total Tests Taken: {total_tests}\n"
        report += f"Average Score: {avg_score:.1f}%\n"
        report += "\nDetailed Test History:\n"
        for test in st.session_state.test_history:
            report += f"\nCourse: {test['course']}\n"
            report += f"Date: {test['date']}\n"
            report += f"Score: {test['score']:.1f}%\n"
            report += f"Duration: {test['duration']} seconds\n"
            report += f"Correct Answers: {test['correct_answers']}/{test['total_questions']}\n"
        
        # Create download button
        st.download_button(
            label="Download Report",
            data=report,
            file_name="test_report.txt",
            mime="text/plain"
        )

def display_reports():
    if not st.session_state.logged_in:
        st.warning("Please login first")
        return
        
    candidate_data = st.session_state.user_data
    st.title("Reports")

    # Sample data (would come from database in production)
    courses = {
        "HTML": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-1.png"},
        "CSS": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-2.png"},
        "JavaScript": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-3.png"},
        "Bootstrap": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-4.png"},
        "jQuery": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-5.png"},
        "SASS": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-6.png"},
        "PHP": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-7.png"},
        "MySQL": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-8.png"},
        "React": {"tutor": "John Deo", "videos": 10, "thumb": "thumb-9.png"}
    }

    progress = {
        "HTML": 8,
        "CSS": 6,
        "JavaScript": 4,
        "Bootstrap": 0,
        "jQuery": 0,
        "SASS": 0,
        "PHP": 0,
        "MySQL": 0,
        "React": 0
    }

    # MCQ and Project Scores (would come from database)
    mcq_scores = {
        "HTML": 85,
        "CSS": 92,
        "JavaScript": 78,
        "Bootstrap": 0,
        "jQuery": 0,
        "SASS": 0,
        "PHP": 0,
        "MySQL": 0,
        "React": 0
    }
    
    project_scores = {
        "HTML": 90,
        "CSS": 88,
        "JavaScript": 82,
        "Bootstrap": 0,
        "jQuery": 0,
        "SASS": 0,
        "PHP": 0,
        "MySQL": 0,
        "React": 0
    }

    # Individual Candidate Report
    st.subheader("Individual Candidate Report")
    st.write(f"Name: {candidate_data[1]}")
    st.write(f"Email: {candidate_data[2]}")
    st.write(f"Degree: {candidate_data[4]}")
    st.write(f"Specialization: {candidate_data[5]}")

    st.subheader("Training Progress")
    for course, completed in progress.items():
        total_videos = courses[course]["videos"]
        completion_percentage = (completed / total_videos) * 100
        status = "Completed" if completed == total_videos else "In Progress"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(course)
        with col2:
            st.write(f"{completed}/{total_videos} videos")
        with col3:
            st.write(f"{completion_percentage:.1f}% - {status}")

    st.subheader("Assessment Scores")
    for course, score in mcq_scores.items():
        col1, col2 = st.columns(2)
        with col1:
            st.write(course)
        with col2:
            st.write(f"MCQ Score: {score}/100")

    for course, score in project_scores.items():
        col1, col2 = st.columns(2)
        with col1:
            st.write(course)
        with col2:
            st.write(f"Project Score: {score}/100")

    # Batch-wise Scorecard
    st.subheader("Batch Scorecard")
    batch_data = [
        {"batch": "Java", "total_candidates": 28, "avg_mcq_score": 85, "avg_project_score": 88, "completion_rate": 92},
        {"batch": ".NET", "total_candidates": 26, "avg_mcq_score": 81, "avg_project_score": 84, "completion_rate": 88},
        {"batch": "Data Engineering", "total_candidates": 30, "avg_mcq_score": 90, "avg_project_score": 92, "completion_rate": 95}
    ]

    for batch in batch_data:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(batch["batch"])
        with col2:
            st.write(f"Candidates: {batch['total_candidates']}")
        with col3:
            st.write(f"MCQ Avg: {batch['avg_mcq_score']}/100")
        with col4:
            st.write(f"Project Avg: {batch['avg_project_score']}/100")
            st.write(f"Completion: {batch['completion_rate']}%")

    # College-wise Scorecard
    st.subheader("College Scorecard")
    college_data = [
        {"college": "University A", "total_candidates": 12, "avg_mcq_score": 88, "avg_project_score": 90, "completion_rate": 92},
        {"college": "University B", "total_candidates": 15, "avg_mcq_score": 82, "avg_project_score": 86, "completion_rate": 88},
        {"college": "University C", "total_candidates": 18, "avg_mcq_score": 91, "avg_project_score": 93, "completion_rate": 95}
    ]

    for college in college_data:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(college["college"])
        with col2:
            st.write(f"Candidates: {college['total_candidates']}")
        with col3:
            st.write(f"MCQ Avg: {college['avg_mcq_score']}/100")
        with col4:
            st.write(f"Project Avg: {college['avg_project_score']}/100")
            st.write(f"Completion: {college['completion_rate']}%")

def display_analysis():
    if not st.session_state.logged_in:
        st.warning("Please login first")
        return
        
    st.title("Analysis")
    
    # Fetch candidate data and training progress
    candidate_data = st.session_state.user_data
    progress = {
        "HTML": 8,
        "CSS": 6,
        "JavaScript": 4,
        "Bootstrap": 0,
        "jQuery": 0,
        "SASS": 0,
        "PHP": 0,
        "MySQL": 0,
        "React": 0
    }
    
    # Performance Trends
    st.subheader("Performance Trends")
    
    # Plot progress over time
    progress_over_time = pd.DataFrame({
        'Course': list(progress.keys()),
        'Completed': list(progress.values())
    })
    
    fig = px.line(progress_over_time, x='Course', y='Completed', title='Progress Over Time')
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation between certifications/internships and performance
    st.subheader("Certifications and Internships Impact")
    
    # Sample data (would come from database in production)
    candidate_certs = ["AWS", "Azure", "NPTEL"]
    candidate_internships = ["Acme Inc.", "XYZ Corp."]
    
    # Analyze impact of certifications and internships
    st.write(f"Certifications: {', '.join(candidate_certs)}")
    st.write(f"Internships: {', '.join(candidate_internships)}")
    
    # Add analysis and insights here
    st.info("Further analysis on the impact of certifications and internships is coming soon.")
    
    # Programming Language Impact
    st.subheader("Programming Language Impact")
    
    # Sample data (would come from database in production)
    candidate_languages = ["Python", "Java", "C#", "JavaScript"]
    
    # Analyze impact of programming languages
    st.write(f"Known Programming Languages: {', '.join(candidate_languages)}")
    
    # Add analysis and insights here
    st.info("Further analysis on the impact of programming languages is coming soon.")

def display_internship_details():
    if not st.session_state.logged_in:
        st.warning("Please login first")
        return
        
    candidate_data = st.session_state.user_data
    st.title("Internship Details")
    
    # Sample internship data (would come from database in production)
    internships = [
        {
            "company": "Acme Inc.",
            "duration": "3 months",
            "description": "Worked as a software engineering intern, developing a web application using React and Node.js.",
            "certificate": "acme_internship_certificate.pdf"
        },
        {
            "company": "XYZ Corp.",
            "duration": "2 months",
            "description": "Completed a data engineering internship, working with Python, Hadoop, and Spark.",
            "certificate": "xyz_internship_certificate.pdf"
        }
    ]
    
    # Display internship details
    for internship in internships:
        with st.expander(internship["company"]):
            st.write(f"Duration: {internship['duration']}")
            st.write(f"Description: {internship['description']}")
            
            # Display certificate if available
            if internship["certificate"]:
                certificate_file = open(internship["certificate"], "rb")
                st.download_button(
                    label="Download Certificate",
                    data=certificate_file,
                    file_name=internship["certificate"],
                    mime="application/pdf"
                )
            else:
                st.write("No certificate available.")

def main():
    initialize_session_state()

    st.sidebar.title("Navigation")
    
    if st.session_state.logged_in:
        # Create a dictionary mapping page names to their corresponding functions
        pages = {
            "Home": candidate_dashboard,
            "Training Progress": display_training_progress,
            "Feedback": display_feedback,
            "MCQ Tests": display_mcq_test,
            "Test Reports": display_test_reports,
            "Reports": display_reports,
            "Analysis": display_analysis,
            "Internship Details": display_internship_details
        }
        
        # Store the selection in session state to maintain it across reruns
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Home"
        
        # Create the radio selector
        selection = st.sidebar.radio(
            "Go to",
            list(pages.keys()),
            key="navigation",
            index=list(pages.keys()).index(st.session_state.current_page)
        )
        
        # Update current page in session state
        st.session_state.current_page = selection
        
        # Add logout button
        st.sidebar.button("Logout", on_click=logout)
        
        # Display the selected page
        pages[selection]()
        
    else:
        # Login page handling
        pages = {
            "Candidate Login": candidate_login
        }
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        pages[selection]()

if __name__ == "__main__":
    main()