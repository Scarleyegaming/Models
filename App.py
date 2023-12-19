
import streamlit as st
from transcribe import Transcribe
from Summarizer import summarizer
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')
import google.generativeai as genai
genai.configure(api_key="AIzaSyBBqGd1wQcmGC9USmMEm87rUZpe63OM6AE")

st.set_page_config(page_title="Optimind Medical Bot",page_icon=":smile",layout="centered",initial_sidebar_state="collapsed",menu_items={"Get Help":"mailto:Mvihnp@gmail.com"})
st.markdown("""
# Optimind Medical Bot 
> A bot which creates medical notes using just your audio sessions.  
> It's **AI** is so powerfull it creates superb **ready-to-use** Medical Note.  
> Use It Quick And Easy.
""")


upload_audio =st.file_uploader(label="Upload Your Audio File",type=["wav","mp3","flacc","m4a"])
if upload_audio is not None:
    
    if st.button("Start"):
        
        file = upload_audio
        save_folder = './temp'
        save_path = Path(save_folder, file.name)
        with open(save_path, mode='wb') as w:
           w.write(file.getvalue())

        
        with st.status("Transcribing Your Audio..."):
            transcribe = Transcribe(f"./temp/{file.name}")
        
        
        os.remove(f"./temp/{file.name}")
        with st.status("Summarizing Your Text...."):
            summarizedtext = summarizer(transcribe)
        prompt=f"Generaten a mental soap note using this text:{summarizedtext}"
        with st.status("Creating Notes...."):
          model = genai.GenerativeModel('gemini-pro')
          response =  model.generate_content(f"create a mental soap note using this text:{summarizedtext}")
          st.code(f"{response}","python")

with st.sidebar:
    st.markdown("""
    # `How To Use!`        
    > 1. `Upload Your Audio`
    > 2. `Click Start Button`
    > 3. `Wait For Audio To Transcribed`
    > 4. `Wait For The Transcription To Summarized`
    > 5. `Wait For AI To Create Your Medical Note`
    > 6. `Valla Your All Ready To Go`
    > 7. **`Use Copy Button To Copy`**
""")
   






