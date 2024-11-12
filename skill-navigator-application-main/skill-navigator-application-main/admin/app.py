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

# Initialize Gemini Pro
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

# Database connection
def get_database_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="skill_navigator"
    )
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="skill_navigator"
)
cursor = db.cursor(buffered=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Sidebar navigation
def sidebar_navigation():
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        ["Home", "Candidate Registration", "Batch Allocation", "Training Progress", 
         "Feedback", "Reports", "Analysis", "Internship Details"]
    )
def hash_password(password):
    """Hash a password for storing."""
    # Convert the password to bytes
    password_bytes = password.encode('utf-8')
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    # Return the hashed password as a string
    return hashed.decode('utf-8')

def candidate_registration():
    st.title("Candidate Registration")

    with st.form("registration_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        degree = st.text_input("Degree")
        specialization = st.text_input("Specialization")
        phone_number = st.text_input("Phone Number")
        linkedin_profile = st.text_input("LinkedIn Profile")
        github_profile = st.text_input("GitHub Profile")
        certifications = st.text_input("certifications")
        internships = st.text_input("internships")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Register")

        if submit_button:
            if not all([name, email, password]):
                st.error("Please fill in all required fields (Name, Email, and Password)")
                return

            try:
                # Hash password
                hashed_password = hash_password(password)
                
                # Insert into database
                sql = """INSERT INTO candidates 
                         (name, email, hashed_password, degree, specialization, 
                          phone_number, linkedin_profile, github_profile, certifications, internships) 
                         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                values = (name, email, hashed_password, degree, specialization, 
                         phone_number, linkedin_profile, github_profile, certifications, internships)
                
                cursor.execute(sql, values)
                db.commit()
                st.success("Registration successful! Please proceed to login.")
            except mysql.connector.Error as err:
                if err.errno == 1062:  # Duplicate entry error
                    st.error("This email is already registered. Please use a different email or login.")
                else:
                    st.error(f"An error occurred: {err}")
                db.rollback()

# Enhanced batch allocation with automatic and manual options
def batch_allocation():
    st.header("Batch Allocation")
    
    allocation_type = st.radio("Allocation Type", ["Automatic", "Manual"])
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    if allocation_type == "Automatic":
        perform_automatic_allocation(cursor, conn)
    else:
        perform_manual_allocation(cursor, conn)
    
    # Display current batch allocations
    display_batch_allocations(cursor)
    
    conn.close()

def perform_automatic_allocation(cursor, conn):
    # Get unallocated candidates
    cursor.execute("""
        SELECT c.candidate_id, c.name, c.email, GROUP_CONCAT(cert_name) as certifications
        FROM candidates c
        LEFT JOIN batch_allocations ba ON c.candidate_id = ba.candidate_id
        LEFT JOIN certifications cert ON c.candidate_id = cert.candidate_id
        WHERE ba.candidate_id IS NULL
        GROUP BY c.candidate_id
    """)
    
    unallocated = cursor.fetchall()
    
    if unallocated:
        st.subheader("Processing Unallocated Candidates")
        progress_bar = st.progress(0)
        
        for idx, candidate in enumerate(unallocated):
            certifications = candidate[3].lower() if candidate[3] else ""
            
            # Determine batch type based on certifications
            batch_type = None
            if any(cert in certifications for cert in ['aws', 'java']):
                batch_type = 'Java'
            elif any(cert in certifications for cert in ['azure', '.net']):
                batch_type = '.NET'
            elif 'python' in certifications:
                batch_type = 'Data Engineering'
            
            if batch_type:
                # Find available batch
                cursor.execute("""
                    SELECT b.batch_id
                    FROM batches b
                    LEFT JOIN batch_allocations ba ON b.batch_id = ba.batch_id
                    WHERE b.batch_type = %s
                    GROUP BY b.batch_id
                    HAVING COUNT(ba.allocation_id) < b.max_capacity
                    LIMIT 1
                """, (batch_type,))
                
                available_batch = cursor.fetchone()
                
                if available_batch:
                    cursor.execute("""
                        INSERT INTO batch_allocations (candidate_id, batch_id)
                        VALUES (%s, %s)
                    """, (candidate[0], available_batch[0]))
                    
                    conn.commit()
                    st.success(f"✅ Allocated {candidate[1]} to {batch_type} batch")
            
            progress_bar.progress((idx + 1) / len(unallocated))

def perform_manual_allocation(cursor, conn):
    # Get unallocated candidates
    cursor.execute("""
        SELECT c.candidate_id, c.name
        FROM candidates c
        LEFT JOIN batch_allocations ba ON c.candidate_id = ba.candidate_id
        WHERE ba.candidate_id IS NULL
    """)
    unallocated = cursor.fetchall()
    
    # Get available batches
    cursor.execute("SELECT batch_id, batch_name, batch_type FROM batches")
    batches = cursor.fetchall()
    
    if unallocated and batches:
        selected_candidate = st.selectbox(
            "Select Candidate",
            unallocated,
            format_func=lambda x: x[1]
        )
        
        selected_batch = st.selectbox(
            "Select Batch",
            batches,
            format_func=lambda x: f"{x[1]} ({x[2]})"
        )
        
        if st.button("Allocate"):
            cursor.execute("""
                INSERT INTO batch_allocations (candidate_id, batch_id)
                VALUES (%s, %s)
            """, (selected_candidate[0], selected_batch[0]))
            
            conn.commit()
            st.success(f"Manually allocated {selected_candidate[1]} to {selected_batch[1]}")

def display_batch_allocations(cursor):
    st.subheader("Current Batch Allocations")
    
    cursor.execute("""
        SELECT b.batch_name, b.batch_type, COUNT(ba.candidate_id) as current_count,
               b.max_capacity
        FROM batches b
        LEFT JOIN batch_allocations ba ON b.batch_id = ba.batch_id
        GROUP BY b.batch_id
    """)
    
    allocations = cursor.fetchall()
    
    if allocations:
        fig = go.Figure()
        
        for batch in allocations:
            fig.add_trace(go.Bar(
                name=batch[0],
                x=[batch[0]],
                y=[batch[2]],
                text=f"{batch[2]}/{batch[3]}",
                textposition='auto',
            ))
            
            # Add capacity line
            fig.add_shape(
                type="line",
                x0=batch[0],
                y0=batch[3],
                x1=batch[0],
                y1=batch[3],
                line=dict(color="red", width=2, dash="dash")
            )
        
        fig.update_layout(
            title="Batch Allocation Status",
            xaxis_title="Batch",
            yaxis_title="Number of Candidates",
            showlegend=False
        )
        
        st.plotly_chart(fig)

# Enhanced training progress tracking
def training_progress():
    st.header("Training Progress")
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    # Get all candidates with their batch information
    cursor.execute("""
        SELECT c.candidate_id, c.name, b.batch_name, b.batch_type
        FROM candidates c
        JOIN batch_allocations ba ON c.candidate_id = ba.candidate_id
        JOIN batches b ON ba.batch_id = b.batch_id
        ORDER BY b.batch_name, c.name
    """)
    
    candidates = cursor.fetchall()
    
    if candidates:
        # Batch filter
        batch_names = list(set(c[2] for c in candidates))
        selected_batch = st.selectbox("Filter by Batch", ["All"] + batch_names)
        
        filtered_candidates = candidates
        if selected_batch != "All":
            filtered_candidates = [c for c in candidates if c[2] == selected_batch]
        
        selected_candidate = st.selectbox(
            "Select Candidate",
            filtered_candidates,
            format_func=lambda x: f"{x[1]} ({x[2]})"
        )
        
        if selected_candidate:
            # Display current progress
            display_current_progress(cursor, selected_candidate[0])
            
            # Update progress form
            st.subheader("Update Progress")
            with st.form("progress_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    topic = st.text_input("Topic")
                    completion = st.slider("Completion Percentage", 0, 100, 0)
                
                with col2:
                    mcq_score = st.number_input("MCQ Score", 0, 100)
                    project_score = st.number_input("Project Score", 0, 100)
                
                submitted = st.form_submit_button("Update Progress")
                
                if submitted:
                    cursor.execute("""
                        INSERT INTO training_progress 
                        (candidate_id, topic, completion_percentage, mcq_score, project_score)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (selected_candidate[0], topic, completion, mcq_score, project_score))
                    
                    conn.commit()
                    st.success("Progress updated successfully!")
                    st.experimental_rerun()
    
    conn.close()

def display_current_progress(cursor, candidate_id):
    cursor.execute("""
        SELECT topic, completion_percentage, mcq_score, project_score, 
               DATE_FORMAT(updated_at, '%Y-%m-%d %H:%i') as update_time
        FROM training_progress
        WHERE candidate_id = %s
        ORDER BY updated_at DESC
    """, (candidate_id,))
    
    progress_data = cursor.fetchall()
    
    if progress_data:
        progress_df = pd.DataFrame(progress_data, 
                                 columns=["Topic", "Completion %", "MCQ Score", 
                                        "Project Score", "Last Updated"])
        
        # Display progress table
        st.dataframe(progress_df)
        
        # Create progress visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=progress_df["Topic"],
            y=progress_df["MCQ Score"],
            name="MCQ Score",
            mode='lines+markers'
        ))
        
        fig.add_trace(go.Scatter(
            x=progress_df["Topic"],
            y=progress_df["Project Score"],
            name="Project Score",
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title="Progress Tracking",
            xaxis_title="Topics",
            yaxis_title="Scores",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)

# Enhanced feedback collection with sentiment analysis
def feedback_collection():
    st.header("Feedback Collection")
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    # Get all candidates with their batch information
    cursor.execute("""
        SELECT c.candidate_id, c.name, b.batch_name
        FROM candidates c
        JOIN batch_allocations ba ON c.candidate_id = ba.candidate_id
        JOIN batches b ON ba.batch_id = b.batch_id
        ORDER BY b.batch_name, c.name
    """)
    
    candidates = cursor.fetchall()
    
    if candidates:
        # Batch filter for feedback
        batch_names = list(set(c[2] for c in candidates))
        selected_batch = st.selectbox("Filter by Batch", ["All"] + batch_names, key="feedback_batch")
        
        filtered_candidates = candidates
        if selected_batch != "All":
            filtered_candidates = [c for c in candidates if c[2] == selected_batch]
        
        selected_candidate = st.selectbox(
            "Select Candidate",
            filtered_candidates,
            format_func=lambda x: f"{x[1]} ({x[2]})",
            key="feedback_candidate"
        )
        
        if selected_candidate:
            # Display previous feedback
            display_previous_feedback(cursor, selected_candidate[0])
            
            # Feedback form
            with st.form("feedback_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    topic = st.text_input("Topic")
                    rating = st.slider("Rating", 1, 5, 3)
                
                with col2:
                    comments = st.text_area("Comments")
                
                submitted = st.form_submit_button("Submit Feedback")
                
                if submitted:
                    # Analyze sentiment using Gemini Pro
                    sentiment_prompt = PromptTemplate(
                        input_variables=["feedback"],
                        template="Analyze the sentiment of this feedback and provide a score between -1 and 1: {feedback}"
                    )
                    sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt)
                    sentiment_result = sentiment_chain.run(comments)
                    
                    try:
                        sentiment_score = float(sentiment_result)
                    except ValueError:
                        sentiment_score = 0.0
                    
                    cursor.execute("""
                        INSERT INTO feedback 
                        (candidate_id, topic, rating, comments, sentiment_score)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (selected_candidate[0], topic, rating, comments, sentiment_score))
                    
                    conn.commit()
                    st.success("Feedback submitted successfully!")
                    st.experimental_rerun()
    
    conn.close()

def display_previous_feedback(cursor, candidate_id):
    cursor.execute("""
        SELECT topic, rating, comments, sentiment_score,
               DATE_FORMAT(feedback_date, '%Y-%m-%d %H:%i') as feedback_time
        FROM feedback
        WHERE candidate_id = %s
        ORDER BY feedback_date DESC
    """, (candidate_id,))
    
    feedback_data = cursor.fetchall()
    
    if feedback_data:
        st.subheader("Previous Feedback")
        
        feedback_df = pd.DataFrame(feedback_data,
                                 columns=["Topic", "Rating", "Comments", 
                                        "Sentiment Score", "Feedback Date"])
        
        # Display feedback metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Rating", f"{feedback_df['Rating'].mean():.1f}/5")
        with col2:
            st.metric("Total Feedback Count", len(feedback_df))
        with col3:
            st.metric("Average Sentiment", f"{feedback_df['Sentiment Score'].mean():.2f}")
        
        # Display feedback table
        st.dataframe(feedback_df)
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=feedback_df["Feedback Date"],
            y=feedback_df["Rating"],
            name="Rating",
            mode='lines+markers'
        ))
        
        fig.add_trace(go.Scatter(
            x=feedback_df["Feedback Date"],
            y=feedback_df["Sentiment Score"],
            name="Sentiment",
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title="Feedback Trends",
            xaxis_title="Date",
            yaxis_title="Score",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)

# Enhanced reports generation
def generate_reports():
    st.header("Reports")
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Individual Candidate", "Batch-wise Scorecard", "College-wise Scorecard", 
         "Topper List", "Batch Comparison", "Feedback Analysis"]
    )
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    if report_type == "Individual Candidate":
        generate_individual_report(cursor)
    elif report_type == "Batch-wise Scorecard":
        generate_batch_scorecard(cursor)
    elif report_type == "College-wise Scorecard":
        generate_college_scorecard(cursor) # type: ignore
    elif report_type == "Topper List":
        generate_topper_list(cursor) # type: ignore
    elif report_type == "Batch Comparison":
        generate_batch_comparison(cursor) # type: ignore
    elif report_type == "Feedback Analysis":
        generate_feedback_analysis(cursor) # type: ignore
    
    conn.close()

def generate_individual_report(cursor):
    cursor.execute("""
        SELECT 
            c.candidate_id,
            c.name,
            c.email,
            c.degree,
            b.batch_name,
            GROUP_CONCAT(DISTINCT pl.language_name) as languages,
            COUNT(DISTINCT cert.cert_id) as certification_count,
            COUNT(DISTINCT i.internship_id) as internship_count,
            AVG(tp.completion_percentage) as avg_completion,
            AVG(tp.mcq_score) as avg_mcq,
            AVG(tp.project_score) as avg_project,
            AVG(f.rating) as avg_feedback
        FROM candidates c
        LEFT JOIN batch_allocations ba ON c.candidate_id = ba.candidate_id
        LEFT JOIN batches b ON ba.batch_id = b.batch_id
        LEFT JOIN programming_languages pl ON c.candidate_id = pl.candidate_id
        LEFT JOIN certifications cert ON c.candidate_id = cert.candidate_id
        LEFT JOIN internships i ON c.candidate_id = i.candidate_id
        LEFT JOIN training_progress tp ON c.candidate_id = tp.candidate_id
        LEFT JOIN feedback f ON c.candidate_id = f.candidate_id
        GROUP BY c.candidate_id
    """)
    
    data = cursor.fetchall()
    
    if data:
        df = pd.DataFrame(data, columns=[
            "ID", "Name", "Email", "Degree", "Batch", "Programming Languages",
            "Certifications", "Internships", "Avg Completion %", 
            "Avg MCQ Score", "Avg Project Score", "Avg Feedback"
        ])
        
        # Select candidate for detailed report
        selected_candidate = st.selectbox(
            "Select Candidate",
            df["Name"].unique()
        )
        
        candidate_data = df[df["Name"] == selected_candidate].iloc[0]
        
        # Display candidate details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            st.write(f"**Name:** {candidate_data['Name']}")
            st.write(f"**Email:** {candidate_data['Email']}")
            st.write(f"**Degree:** {candidate_data['Degree']}")
            st.write(f"**Batch:** {candidate_data['Batch']}")
        
        with col2:
            st.subheader("Skills & Achievements")
            st.write(f"**Programming Languages:** {candidate_data['Programming Languages']}")
            st.write(f"**Certifications:** {int(candidate_data['Certifications'])}")
            st.write(f"**Internships:** {int(candidate_data['Internships'])}")
        
        # Performance metrics
        st.subheader("Performance Metrics")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Avg Completion", f"{candidate_data['Avg Completion %']:.1f}%")
        with metrics_col2:
            st.metric("Avg MCQ Score", f"{candidate_data['Avg MCQ Score']:.1f}")
        with metrics_col3:
            st.metric("Avg Project Score", f"{candidate_data['Avg Project Score']:.1f}")
        with metrics_col4:
            st.metric("Avg Feedback", f"{candidate_data['Avg Feedback']:.1f}/5")
        
        # Get detailed progress data
        cursor.execute("""
            SELECT topic, completion_percentage, mcq_score, project_score,
                   DATE_FORMAT(updated_at, '%Y-%m-%d') as date
            FROM training_progress
            WHERE candidate_id = %s
            ORDER BY updated_at
        """, (candidate_data['ID'],))
        
        progress_data = cursor.fetchall()
        
        if progress_data:
            progress_df = pd.DataFrame(progress_data,
                                     columns=["Topic", "Completion %", 
                                            "MCQ Score", "Project Score", "Date"])
            
            # Progress visualization
            st.subheader("Progress Timeline")
            fig = px.line(progress_df, x="Date", y=["MCQ Score", "Project Score"],
                         title="Performance Trends")
            st.plotly_chart(fig)
            
            # Detailed progress table
            st.subheader("Detailed Progress")
            st.dataframe(progress_df)

def generate_batch_scorecard(cursor):
    cursor.execute("""
        SELECT 
            b.batch_name,
            b.batch_type,
            COUNT(DISTINCT c.candidate_id) as total_candidates,
            AVG(tp.completion_percentage) as avg_completion,
            AVG(tp.mcq_score) as avg_mcq,
            AVG(tp.project_score) as avg_project,
            AVG(f.rating) as avg_feedback
        FROM batches b
        LEFT JOIN batch_allocations ba ON b.batch_id = ba.batch_id
        LEFT JOIN candidates c ON ba.candidate_id = c.candidate_id
        LEFT JOIN training_progress tp ON c.candidate_id = tp.candidate_id
        LEFT JOIN feedback f ON c.candidate_id = f.candidate_id
        GROUP BY b.batch_id
    """)
    
    data = cursor.fetchall()
    
    if data:
        df = pd.DataFrame(data, columns=[
            "Batch Name", "Batch Type", "Total Candidates",
            "Avg Completion %", "Avg MCQ Score", 
            "Avg Project Score", "Avg Feedback"
        ])
        
        # Display batch metrics
        st.subheader("Batch Performance Overview")
        st.dataframe(df)
        
        # Visualization
        fig = px.bar(df, x="Batch Name", 
                    y=["Avg MCQ Score", "Avg Project Score", "Avg Feedback"],
                    title="Batch Performance Comparison",
                    barmode="group")
        st.plotly_chart(fig)
        
        # Detailed batch analysis
        st.subheader("Detailed Batch Analysis")
        selected_batch = st.selectbox("Select Batch", df["Batch Name"].unique())
        
        cursor.execute("""
            SELECT 
                c.name,
                AVG(tp.completion_percentage) as avg_completion,
                AVG(tp.mcq_score) as avg_mcq,
                AVG(tp.project_score) as avg_project
            FROM candidates c
            JOIN batch_allocations ba ON c.candidate_id = ba.candidate_id
            JOIN batches b ON ba.batch_id = b.batch_id
            LEFT JOIN training_progress tp ON c.candidate_id = tp.candidate_id
            WHERE b.batch_name = %s
            GROUP BY c.candidate_id
            ORDER BY avg_mcq DESC
        """, (selected_batch,))
        
        batch_data = cursor.fetchall()
        
        if batch_data:
            batch_df = pd.DataFrame(batch_data, columns=[
                "Candidate", "Avg Completion %", 
                "Avg MCQ Score", "Avg Project Score"
            ])
            
            st.dataframe(batch_df)

def generate_individual_report(cursor):
    cursor.execute("""
        SELECT 
            c.candidate_id,
            c.name,
            c.email,
            c.degree,
            b.batch_name,
            GROUP_CONCAT(DISTINCT pl.language_name) as languages,
            COUNT(DISTINCT cert.cert_id) as certification_count,
            COUNT(DISTINCT i.internship_id) as internship_count,
            COALESCE(AVG(tp.completion_percentage), 0) as avg_completion,
            COALESCE(AVG(tp.mcq_score), 0) as avg_mcq,
            COALESCE(AVG(tp.project_score), 0) as avg_project,
            COALESCE(AVG(f.rating), 0) as avg_feedback
        FROM candidates c
        LEFT JOIN batch_allocations ba ON c.candidate_id = ba.candidate_id
        LEFT JOIN batches b ON ba.batch_id = b.batch_id
        LEFT JOIN programming_languages pl ON c.candidate_id = pl.candidate_id
        LEFT JOIN certifications cert ON c.candidate_id = cert.candidate_id
        LEFT JOIN internships i ON c.candidate_id = i.candidate_id
        LEFT JOIN training_progress tp ON c.candidate_id = tp.candidate_id
        LEFT JOIN feedback f ON c.candidate_id = f.candidate_id
        GROUP BY 
            c.candidate_id,
            c.name,
            c.email,
            c.degree,
            b.batch_name
        ORDER BY c.name
    """)
    
    data = cursor.fetchall()
    
    if data:
        df = pd.DataFrame(data, columns=[
            "ID", "Name", "Email", "Degree", "Batch", "Programming Languages",
            "Certifications", "Internships", "Avg Completion %", 
            "Avg MCQ Score", "Avg Project Score", "Avg Feedback"
        ])
        
        # Select candidate for detailed report
        selected_candidate = st.selectbox(
            "Select Candidate",
            df["Name"].unique()
        )
        
        if selected_candidate:  # Add check to prevent IndexError
            candidate_data = df[df["Name"] == selected_candidate].iloc[0]
            
            # Display candidate details
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Basic Information")
                st.write(f"**Name:** {candidate_data['Name']}")
                st.write(f"**Email:** {candidate_data['Email']}")
                st.write(f"**Degree:** {candidate_data['Degree']}")
                st.write(f"**Batch:** {candidate_data['Batch'] if pd.notna(candidate_data['Batch']) else 'Not Assigned'}")
            
            with col2:
                st.subheader("Skills & Achievements")
                st.write(f"**Programming Languages:** {candidate_data['Programming Languages'] if pd.notna(candidate_data['Programming Languages']) else 'None'}")
                st.write(f"**Certifications:** {int(candidate_data['Certifications'])}")
                st.write(f"**Internships:** {int(candidate_data['Internships'])}")
            
            # Performance metrics
            st.subheader("Performance Metrics")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Avg Completion", f"{candidate_data['Avg Completion %']:.1f}%")
            with metrics_col2:
                st.metric("Avg MCQ Score", f"{candidate_data['Avg MCQ Score']:.1f}")
            with metrics_col3:
                st.metric("Avg Project Score", f"{candidate_data['Avg Project Score']:.1f}")
            with metrics_col4:
                st.metric("Avg Feedback", f"{candidate_data['Avg Feedback']:.1f}/5")
            
            # Get detailed progress data
            cursor.execute("""
                SELECT topic, completion_percentage, mcq_score, project_score,
                       DATE_FORMAT(updated_at, '%Y-%m-%d') as date
                FROM training_progress
                WHERE candidate_id = %s
                ORDER BY updated_at
            """, (candidate_data['ID'],))
            
            progress_data = cursor.fetchall()
            
            if progress_data:
                progress_df = pd.DataFrame(progress_data,
                                         columns=["Topic", "Completion %", 
                                                "MCQ Score", "Project Score", "Date"])
                
                # Progress visualization
                st.subheader("Progress Timeline")
                fig = px.line(progress_df, x="Date", y=["MCQ Score", "Project Score"],
                             title="Performance Trends")
                st.plotly_chart(fig)
                
                # Detailed progress table
                st.subheader("Detailed Progress")
                st.dataframe(progress_df)
    else:
        st.info("No candidate data available.")

def analyze_performance_trends(cursor):
    st.subheader("Performance Trends Analysis")
    
    # Get performance data over time
    cursor.execute("""
        SELECT 
            DATE_FORMAT(tp.updated_at, '%Y-%m') as month,
            AVG(tp.mcq_score) as avg_mcq,
            AVG(tp.project_score) as avg_project,
            COUNT(DISTINCT tp.candidate_id) as candidate_count
        FROM training_progress tp
        GROUP BY DATE_FORMAT(tp.updated_at, '%Y-%m')
        ORDER BY month
    """)
    
    data = cursor.fetchall()
    
    if data:
        df = pd.DataFrame(data, columns=[
            "Month", "Avg MCQ Score", "Avg Project Score", "Candidate Count"
        ])
        
        # Performance trends visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df["Month"],
            y=df["Avg MCQ Score"],
            name="MCQ Score",
            mode='lines+markers'
        ))
        
        fig.add_trace(go.Scatter(
            x=df["Month"],
            y=df["Avg Project Score"],
            name="Project Score",
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title="Average Performance Trends Over Time",
            xaxis_title="Month",
            yaxis_title="Score",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)
        
        # Topic-wise performance analysis
        st.subheader("Topic-wise Performance")
        cursor.execute("""
            SELECT 
                topic,
                AVG(mcq_score) as avg_mcq,
                AVG(project_score) as avg_project,
                COUNT(DISTINCT candidate_id) as attempt_count
            FROM training_progress
            GROUP BY topic
            ORDER BY avg_mcq DESC
        """)
        
        topic_data = cursor.fetchall()
        if topic_data:
            topic_df = pd.DataFrame(topic_data, columns=[
                "Topic", "Avg MCQ Score", "Avg Project Score", "Attempts"
            ])
            
            st.dataframe(topic_df)
    else:
        st.info("No performance data available for analysis.")

def analyze_feedback(cursor):
    st.subheader("Feedback Analysis")
    
    # Get feedback trends over time
    cursor.execute("""
        SELECT 
            DATE_FORMAT(feedback_date, '%Y-%m') as month,
            AVG(rating) as avg_rating,
            AVG(sentiment_score) as avg_sentiment,
            COUNT(*) as feedback_count
        FROM feedback
        GROUP BY DATE_FORMAT(feedback_date, '%Y-%m')
        ORDER BY month
    """)
    
    data = cursor.fetchall()
    
    if data:
        df = pd.DataFrame(data, columns=[
            "Month", "Avg Rating", "Avg Sentiment", "Feedback Count"
        ])
        
        # Feedback trends visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df["Month"],
            y=df["Avg Rating"],
            name="Average Rating",
            mode='lines+markers'
        ))
        
        fig.add_trace(go.Scatter(
            x=df["Month"],
            y=df["Avg Sentiment"],
            name="Sentiment Score",
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title="Feedback Trends Over Time",
            xaxis_title="Month",
            yaxis_title="Score",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)
        
        # Topic-wise feedback analysis
        st.subheader("Topic-wise Feedback")
        cursor.execute("""
            SELECT 
                topic,
                AVG(rating) as avg_rating,
                AVG(sentiment_score) as avg_sentiment,
                COUNT(*) as feedback_count
            FROM feedback
            GROUP BY topic
            ORDER BY avg_rating DESC
        """)
        
        topic_data = cursor.fetchall()
        if topic_data:
            topic_df = pd.DataFrame(topic_data, columns=[
                "Topic", "Avg Rating", "Avg Sentiment", "Feedback Count"
            ])
            
            st.dataframe(topic_df)
    else:
        st.info("No feedback data available for analysis.")

def analyze_certification_impact(cursor):
    st.subheader("Certification Impact Analysis")
    
    # Compare performance between certified and non-certified candidates
    cursor.execute("""
        SELECT 
            CASE 
                WHEN cert.cert_id IS NOT NULL THEN 'Certified'
                ELSE 'Non-Certified'
            END as certification_status,
            AVG(tp.mcq_score) as avg_mcq,
            AVG(tp.project_score) as avg_project,
            COUNT(DISTINCT c.candidate_id) as candidate_count
        FROM candidates c
        LEFT JOIN certifications cert ON c.candidate_id = cert.candidate_id
        LEFT JOIN training_progress tp ON c.candidate_id = tp.candidate_id
        GROUP BY certification_status
    """)
    
    data = cursor.fetchall()
    
    if data:
        df = pd.DataFrame(data, columns=[
            "Status", "Avg MCQ Score", "Avg Project Score", "Candidate Count"
        ])
        
        # Visualization
        fig = px.bar(df, x="Status", 
                    y=["Avg MCQ Score", "Avg Project Score"],
                    title="Performance Comparison by Certification Status",
                    barmode="group")
        
        st.plotly_chart(fig)
        
        # Certification provider analysis
        st.subheader("Performance by Certification Provider")
        cursor.execute("""
            SELECT 
                cert.cert_provider,
                AVG(tp.mcq_score) as avg_mcq,
                AVG(tp.project_score) as avg_project,
                COUNT(DISTINCT c.candidate_id) as candidate_count
            FROM candidates c
            JOIN certifications cert ON c.candidate_id = cert.candidate_id
            LEFT JOIN training_progress tp ON c.candidate_id = tp.candidate_id
            GROUP BY cert.cert_provider
            HAVING cert_provider IS NOT NULL
            ORDER BY avg_mcq DESC
        """)
        
        provider_data = cursor.fetchall()
        if provider_data:
            provider_df = pd.DataFrame(provider_data, columns=[
                "Provider", "Avg MCQ Score", "Avg Project Score", "Candidate Count"
            ])
            
            st.dataframe(provider_df)
    else:
        st.info("No certification data available for analysis.")

def analyze_batch_performance(cursor):
    st.subheader("Batch Performance Analysis")
    
    # Get batch-wise performance metrics
    cursor.execute("""
        SELECT 
            b.batch_name,
            b.batch_type,
            AVG(tp.mcq_score) as avg_mcq,
            AVG(tp.project_score) as avg_project,
            AVG(f.rating) as avg_feedback,
            COUNT(DISTINCT c.candidate_id) as candidate_count
        FROM batches b
        LEFT JOIN batch_allocations ba ON b.batch_id = ba.batch_id
        LEFT JOIN candidates c ON ba.candidate_id = c.candidate_id
        LEFT JOIN training_progress tp ON c.candidate_id = tp.candidate_id
        LEFT JOIN feedback f ON c.candidate_id = f.candidate_id
        GROUP BY b.batch_id, b.batch_name, b.batch_type
    """)
    
    data = cursor.fetchall()
    
    if data:
        df = pd.DataFrame(data, columns=[
            "Batch", "Type", "Avg MCQ Score", "Avg Project Score", 
            "Avg Feedback", "Candidate Count"
        ])
        
        # Batch performance visualization
        fig = px.bar(df, x="Batch", 
                    y=["Avg MCQ Score", "Avg Project Score", "Avg Feedback"],
                    title="Batch Performance Comparison",
                    barmode="group")
        
        st.plotly_chart(fig)
        
        # Batch type analysis
        st.subheader("Performance by Batch Type")
        type_df = df.groupby("Type").agg({
            "Avg MCQ Score": "mean",
            "Avg Project Score": "mean",
            "Avg Feedback": "mean",
            "Candidate Count": "sum"
        }).reset_index()
        
        st.dataframe(type_df)
        
        # Completion rate analysis
        st.subheader("Batch Completion Rate")
        cursor.execute("""
            SELECT 
                b.batch_name,
                COUNT(DISTINCT tp.candidate_id) as completed_candidates,
                COUNT(DISTINCT ba.candidate_id) as total_candidates,
                (COUNT(DISTINCT tp.candidate_id) * 100.0 / 
                 COUNT(DISTINCT ba.candidate_id)) as completion_rate
            FROM batches b
            LEFT JOIN batch_allocations ba ON b.batch_id = ba.batch_id
            LEFT JOIN training_progress tp ON ba.candidate_id = tp.candidate_id
            WHERE tp.completion_percentage >= 100
            GROUP BY b.batch_id, b.batch_name
        """)
        
        completion_data = cursor.fetchall()
        if completion_data:
            completion_df = pd.DataFrame(completion_data, columns=[
                "Batch", "Completed", "Total", "Completion Rate %"
            ])
            
            st.dataframe(completion_df)
    else:
        st.info("No batch performance data available for analysis.")

def main():
    st.set_page_config(
        page_title="Skill Navigator",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Skill Navigator Application")
    
    page = sidebar_navigation()
    
    if page == "Home":
        st.write("Welcome to Skill Navigator Application")
        display_dashboard()
    elif page == "Candidate Registration":
        candidate_registration()
    elif page == "Batch Allocation":
        batch_allocation()
    elif page == "Training Progress":
        training_progress()
    elif page == "Feedback":
        feedback_collection()
    elif page == "Reports":
        generate_reports()
    elif page == "Analysis":
        perform_analysis()

def display_dashboard():
    conn = get_database_connection()
    cursor = conn.cursor()
    
    # Get summary metrics
    cursor.execute("SELECT COUNT(*) FROM candidates")
    total_candidates = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM batch_allocations")
    allocated_candidates = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(mcq_score) FROM training_progress")
    avg_mcq_score = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT AVG(rating) FROM feedback")
    avg_feedback = cursor.fetchone()[0] or 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Candidates", total_candidates)
    with col2:
        st.metric("Allocated Candidates", allocated_candidates)
    with col3:
        st.metric("Avg MCQ Score", f"{avg_mcq_score:.1f}")
    with col4:
        st.metric("Avg Feedback", f"{avg_feedback:.1f}/5")
    
    # Recent activities
    st.subheader("Recent Activities")
    
    # Recent registrations
    cursor.execute("""
        SELECT name, email, DATE_FORMAT(created_at, '%Y-%m-%d %H:%i') as registered_at
        FROM candidates
        ORDER BY created_at DESC
        LIMIT 5
    """)
    recent_registrations = cursor.fetchall()
    
    if recent_registrations:
        st.write("Latest Registrations:")
        recent_reg_df = pd.DataFrame(recent_registrations,
                                   columns=["Name", "Email", "Registered At"])
        st.dataframe(recent_reg_df)
    
    conn.close()

def perform_analysis():
    st.header("Analysis")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Performance Trends", "Feedback Analysis", 
         "Certification Impact", "Batch Performance"]
    )
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    if analysis_type == "Performance Trends":
        analyze_performance_trends(cursor)
    elif analysis_type == "Feedback Analysis":
        analyze_feedback(cursor)
    elif analysis_type == "Certification Impact":
        analyze_certification_impact(cursor)
    elif analysis_type == "Batch Performance":
        analyze_batch_performance(cursor)
    
    conn.close()

if __name__=="__main__":
    main()