from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streamlit import StreamlitCallbackHandler
import streamlit as st
from transcribe import Transcribe
from Summarizer import summarizer
from pathlib import Path
import os
template = """
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. You must complete your answers.
<</SYS>>
{prompt}[/INST]
"""
prompt = PromptTemplate(template=template, input_variables=["prompt"])

st.set_page_config(page_title="Optimind Medical Bot",page_icon=":smile",layout="centered",initial_sidebar_state="collapsed",menu_items={"Get Help":"mailto:Mvihnp@gmail.com"})
st.markdown("""
# Optimind Medical Bot 
> A bot which creates medical notes using just your audio sessions.  
> It's **AI** is so powerfull it creates superb **ready-to-use** Medical Note.  
> Use It Quick And Easy.
""")
container = st.container()
callback_manager = CallbackManager([StreamlitCallbackHandler(container)])

llm = LlamaCpp(
    model_path="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q5_0.gguf",
    temperature=0,
    max_tokens=8000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)



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
        prompt=f"""Generate a Small SOAP Note Under 3075 Words, It Must Contain Subjective, Objective, Assesment And Plan, The Information Must Be In Bullet Points With Only 3 Points In Each Every Point Be Under 10 Words:{summarizedtext}"""
        with st.status("Creating Notes...."):
          response = llm(prompt)
          st.code(f"{response} \n This is only a sample by AI","python")

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
   






