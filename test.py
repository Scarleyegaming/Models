from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
from transcribe import Transcribe
from Summarizer import summarizer
from pathlib import Path
import os
template = """
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}[/INST]
"""
prompt = PromptTemplate(template=template, input_variables=["prompt"])


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="./models/output.bin",
    temperature=0,
    max_tokens=8000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
prompt = f"""Generate a Small SOAP Note Under 3075 Words, It Must Contain Subjective, Objective, Assesment And Plan, The Information Must Be In Bullet Points With Only 3 Points In Each Every Point Be Under 10 Words: Jennifer is a 35-year-old married female with six children. She is a night shift working registered nurse and is going to school to be a family nurse practitioner. She describes feeling constantly overwhelmed and like she needs to get away from society. Some days are worse than others, with her worst days being at an 80% level. She rarely practices self-care. She takes medications for her ADHD and high blood pressure. She has had her gallbladder removed and her tubes tied. She has a good relationship with her husband, although they occasionally get on each other's nerves. She has four children at home, ages 5, 6, 11, and 12. She also has stepchildren who are 18 and 20. She feels tired all the time, even when she is off work. She occasionally has blurry vision and passes out when she changes positions quickly. She was diagnosed with ADHD six months ago and takes Consurda and S-DARS. Her mother may have had ADHD and depression, and her biological father may have had ADHD, while her stepdad may have had bipolar disorder. She would like help addressing her overwhelmed feelings and getting her sleep under control.
                    """

llm(prompt)