import streamlit as st
import subprocess
import os
import zipfile
from io import BytesIO

# Initialize session state for log content
if 'log_content' not in st.session_state or not isinstance(st.session_state.log_content, list):
    st.session_state.log_content = []  # Reset to empty list if not already a list

# Create a placeholder for the log window with custom styling
st.markdown("""
    <style>
        /* Import Fira Code font */
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400&display=swap');
        
        /* Container for the entire terminal */
        div[data-testid="stMarkdownContainer"] > div {
            background-color: transparent !important;
        }
        
        /* Terminal window styling */
        .terminal-window {
            background-color: #000000;
            border: 2px solid #333333;
            border-radius: 5px;
            padding: 16px;
            font-family: 'Fira Code', 'JetBrains Mono', 'SF Mono', 'Menlo', 'Monaco', monospace;
            color: rgba(255, 255, 255, 0.9);
            margin: 10px 0;
            min-height: 300px;
            max-height: 400px;
            overflow-y: auto;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
            position: relative;
        }

        /* Terminal content */
        .terminal-window pre {
            margin: 0;
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: inherit;
        }
    </style>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

# Run button
run_button = st.button("Run")

# Add a title for the terminal section
st.markdown("### Terminal Output")

# Create containers for the terminal
terminal_container = st.container()
log_placeholder = terminal_container.empty()

# Function to append log lines
def append_log(line):
    st.session_state.log_content.append(f"$ {line}")
    # Join all log lines with newlines and display in terminal
    log_content = "\n\n".join(st.session_state.log_content)
    log_placeholder.markdown(
        f'<div class="terminal-window"><pre>{log_content}</pre></div>',
        unsafe_allow_html=True
    )
    # Force the container to re-render at the bottom
    terminal_container.empty()

# Execute scripts and capture logs
if run_button and uploaded_file is not None:
    # Save the uploaded file
    with open("Dimensions_not_registered.xlsx", "wb") as f:
        f.write(uploaded_file.getvalue())
    append_log("File saved successfully.")
    
    # Run data_preprocess.py
    append_log("Running data_preprocess.py...")
    process = subprocess.Popen(['python', 'data_preprocess.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            append_log(output.strip())
    append_log("data_preprocess.py completed.")
    
    # Run autogluon_predictor_ingredient.py
    append_log("Running autogluon_predictor_ingredient.py...")
    process = subprocess.Popen(['python', 'autogluon_predictor_ingredient.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            append_log(output.strip())
    append_log("autogluon_predictor_ingredient.py completed.")
    
    # Run autogluon_predictor_trademark.py
    append_log("Running autogluon_predictor_trademark.py...")
    process = subprocess.Popen(['python', 'autogluon_predictor_trademark.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            append_log(output.strip())
    append_log("autogluon_predictor_trademark.py completed.")
    
    st.success("Processing complete!")

    # List of output files
    output_files = [
        'predictions_high_confidence_ingredient.xlsx',
        'predictions_for_review_ingredient.xlsx',
        'predictions_high_confidence_trademark.xlsx',
        'predictions_for_review_trademark.xlsx',
        'confidence_distribution_summary.xlsx',
        'confidence_distribution_summary_trademark.xlsx'
    ]

    # Check if all files exist
    all_files_exist = all(os.path.exists(file) for file in output_files)

    if all_files_exist:
        # Create a ZIP archive in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for file in output_files:
                zip_file.write(file, arcname=file)
        
        # Provide a download button for the ZIP file
        st.download_button(
            label="Download All Files as ZIP",
            data=zip_buffer.getvalue(),
            file_name="prediction_results.zip",
            mime="application/zip"
        )
    else:
        st.warning("Some output files are missing. Please check the logs for errors.")