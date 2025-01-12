import streamlit as st
import subprocess
import os

# Define CSS style for terminal window
st.markdown(
    '''
    <style>
        .terminal {
            background-color: black;
            color: white;
            padding: 10px;
            overflow-y: auto;
            height: 300px;
            border: 1px solid white;
        }
    </style>
    ''',
    unsafe_allow_html=True
)

# Initialize session state
if 'log_content' not in st.session_state:
    st.session_state.log_content = ""
if 'process_running' not in st.session_state:
    st.session_state.process_running = False

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

# Run button
run_button = st.button("Run")

# Terminal-like window for logs
log_placeholder = st.empty()

# Function to append log lines
def append_log(line):
    st.session_state.log_content += line + "<br>"
    log_placeholder.markdown(f'<pre class="terminal">{st.session_state.log_content}</pre>', unsafe_allow_html=True)

# Function to run scripts and capture logs
def run_scripts():
    st.session_state.process_running = True
    st.session_state.log_content = ""
    append_log("Starting script execution...")

    # Save uploaded file
    if uploaded_file is not None:
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

    st.session_state.process_running = False

# Check if run button is clicked
if run_button and uploaded_file is not None:
    run_scripts()
elif run_button:
    st.warning("Please upload a file before running.")

# Download buttons
if not st.session_state.process_running:
    output_files = [
        'predictions_high_confidence_ingredient.xlsx',
        'predictions_for_review_ingredient.xlsx',
        'predictions_high_confidence_trademark.xlsx',
        'predictions_for_review_trademark.xlsx',
        'confidence_distribution_summary.xlsx',
        'confidence_distribution_summary_trademark.xlsx'
    ]
    for file in output_files:
        if os.path.exists(file):
            with open(file, "rb") as f:
                bytes = f.read()
            st.download_button(
                label=f"Download {file}",
                data=bytes,
                file_name=file,
                mime='application/vnd.ms-excel'
            )
        else:
            st.warning(f"{file} not found.")