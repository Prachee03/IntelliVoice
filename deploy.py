# #importing libraries 
# import streamlit as st
# from indic_transliteration import sanscript
# from indic_transliteration.sanscript import transliterate
# from streamlit_mic_recorder import mic_recorder
# import numpy as np
# import time
# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from langdetect import detect # package for detection of languages (Hindi + Other languages)


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# model_id = "openai/whisper-large-v3"

# @st.cache_resource
# def load_model():
#     model = AutoModelForSpeechSeq2Seq.from_pretrained(
#         model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
#     )
#     model.to(device)
#     return model

# # set page wide 
# st.set_page_config(layout="wide")

# @st.cache_resource
# def load_processor():
#     processor = AutoProcessor.from_pretrained(model_id)
#     return processor

# model = load_model()
# processor = load_processor()

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     max_new_tokens=128,
#     chunk_length_s=30,
#     batch_size=16,
#     return_timestamps=True,
#     torch_dtype=torch_dtype,
#     device=device,
# )

# # Maintain a list to store the last 5 detected results
# @st.cache_resource
# def last_results() :
#     last_5_results = []
#     return last_5_results

# last_5_results = last_results()
# detected_language = None
# detected_text = None
# target_language = None

# audio = None
# languages = {'gu' : 'gujarati',
#              'en' : 'english',
#              'hi' : 'hindi',
#              'bn' : 'bengali',
#              'mr' : 'marathi',
#              'ta' : 'tamil'}

# #Function for detection of language 
# def hindi_to_language(text, detected_language, target_language):
#     if detected_language == "hi" and target_language == "Gujarati": #if detected language is hindi and target language is Gujarati
#         return transliterate(text, sanscript.DEVANAGARI, sanscript.GUJARATI)
#     elif detected_language == "hi" and target_language != "Gujarati": #if detected language is hindi and target language is not Gujarati
#         return text  # Keep the text as Hindi
#     else:
#         # Translate to target language if it's Other language
#         return text

# def main():
#     #st.title("Language Detection from Speech")
#     # st.image('INTELLIVOICE.png',width=200)
#     # st.write("")
    
#     st.title("INTELLIVOICE ")
#     # detected_text=""
#     # transliterated_text=""
#     # hindi_text=""
#     # gujarati_text=""
#     # Splitting the page into two columns
    
    
#     col1, col2 = st.columns(2, gap="large")
#     global detected_language
#     global target_language
#     global detected_text
    
#     with col1:
        
#         target_language = st.selectbox(
#             "Select Target Language:",
#             ("gujarati", 'english', 'hindi', 'bengali', 'marathi', 'tamil'),  # add here other language
#         )
#         audio = mic_recorder(
#             start_prompt="Start recording",
#             stop_prompt="Stop recording",
#             just_once=True,
#             use_container_width=False,
#             callback=None,
#             args=(),
#             kwargs={},
#             key=None,
#         )

#         st.write("OR")
#         uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

#         with st.spinner("Please Wait..."):
#             if audio is not None:
#                 result = pipe(audio["bytes"], generate_kwargs={'language' : target_language})
#                 detected_text = result["text"]
#                 detected_language = detect(detected_text)  # Language detection using langdetect
#                 print(f'DL {detected_language} DT {detected_text}')
                
                
                
                
#             else:
#                 # File upload widget
#                 if uploaded_file is not None:
#                     bytes_data = uploaded_file.getvalue()
#                     result = pipe(bytes_data)
#                     detected_text = result["text"]
#                     detected_language = detect(detected_text)  # Language detection using langdetect
#                     translated_text = hindi_to_language(detected_text, detected_language, target_language)
#                     # st.write("Detected Language:", detected_language)
#                     # st.write("Hindi Text:", detected_text)
#                     # st.write(f"{target_language} Transliteration:", translated_text)
    
#     with col2:
#         st.subheader("Detected Text & Transliteration")
#         print(detected_language)
#         if detected_language is not None:
#             if target_language == languages.get(detected_language) : 
#                 translated_text = hindi_to_language(detected_text, detected_language, target_language)
#                 st.info(f"Detected Language: {detected_language}")
#                 st.info(f"{target_language} Text: {detected_text}")
#                         # output only if input is in selected target language
#                 if target_language == 'gujarati' :
#                     st.write(f"{target_language} Transliteration:", translated_text)
#             else :
#                 st.warning(f"Please speak in {target_language}!!", icon="⚠️")
#                 st.info(f"Detected Language: {detected_language}")
#                 st.info(f"{target_language} Text: {detected_text}")
#                 audio = None
            
# if __name__ == "__main__":
#     main()


#importing libraries 
import streamlit as st
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from streamlit_mic_recorder import mic_recorder
import numpy as np
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langdetect import detect # package for detection of languages (Hindi + Other languages)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"

@st.cache_resource
def load_model():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    return model

# set page wide 
st.set_page_config(layout="wide")

@st.cache_resource
def load_processor():
    processor = AutoProcessor.from_pretrained(model_id)
    return processor

model = load_model()
processor = load_processor()

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Maintain a list to store the last 5 detected results
@st.cache_resource
def last_results() :
    last_5_results = []
    return last_5_results

last_5_results = last_results()
detected_language = None
detected_text = None
target_language = None

audio = None
languages = {'gu' : 'gujarati',
             'en' : 'english',
             'hi' : 'hindi',
             'bn' : 'bengali',
             'mr' : 'marathi',
             'ta' : 'tamil'}

#Function for detection of language 
def hindi_to_language(text, detected_language, target_language):
    if detected_language == "hi" and target_language == "Gujarati": #if detected language is hindi and target language is Gujarati
        return transliterate(text, sanscript.DEVANAGARI, sanscript.GUJARATI)
    elif detected_language == "hi" and target_language != "Gujarati": #if detected language is hindi and target language is not Gujarati
        return text  # Keep the text as Hindi
    else:
        # Translate to target language if it's Other language
        return text

def main():
    #st.title("Language Detection from Speech")
    # st.image('INTELLIVOICE.png',width=200)
    # st.write("")
    
    st.title("INTELLIVOICE ")    
    
    col1, col2 = st.columns(2, gap="large")
    global detected_language
    global target_language
    global detected_text
    
    with col1:
        with st.container(style="border: 1px solid #ddd; border-radius: 4px; padding: 10px; margin: 10px;"):
            target_language = st.selectbox(
                "Select Target Language:",
                ("gujarati", 'english', 'hindi', 'bengali', 'marathi', 'tamil'),  # add here other language
            )
            audio = mic_recorder(
                start_prompt="Start recording",
                stop_prompt="Stop recording",
                just_once=True,
                use_container_width=False,
                callback=None,
                args=(),
                kwargs={},
                key=None,
            )

            st.write("OR")
            uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

            with st.spinner("Please Wait..."):
                if audio is not None:
                    result = pipe(audio["bytes"], generate_kwargs={'language' : target_language})
                    detected_text = result["text"]
                    detected_language = detect(detected_text)  # Language detection using langdetect
                    print(f'DL {detected_language} DT {detected_text}')
                    
                    
                    
                    
                else:
                    # File upload widget
                    if uploaded_file is not None:
                        bytes_data = uploaded_file.getvalue()
                        result = pipe(bytes_data)
                        detected_text = result["text"]
                        detected_language = detect(detected_text)  # Language detection using langdetect
                        translated_text = hindi_to_language(detected_text, detected_language, target_language)
                        # st.write("Detected Language:", detected_language)
                        # st.write("Hindi Text:", detected_text)
                        # st.write(f"{target_language} Transliteration:", translated_text)
    
    with col2:
        with st.container(style="border: 1px solid #ddd; border-radius: 4px; padding: 10px; margin: 10px;"):
            st.subheader("Detected Text & Transliteration")
            print(detected_language)
            if detected_language is not None:
                if target_language == languages.get(detected_language) : 
                    translated_text = hindi_to_language(detected_text, detected_language, target_language)
                    st.info(f"Detected Language: {detected_language}")
                    st.info(f"{target_language} Text: {detected_text}")
                            # output only if input is in selected target language
                    if target_language == 'gujarati' :
                        st.write(f"{target_language} Transliteration:", translated_text)
                else :
                    st.warning(f"Please speak in {target_language}!!", icon="⚠️")
                    st.info(f"Detected Language: {detected_language}")
                    st.info(f"{target_language} Text: {detected_text}")
                    audio = None
            
if __name__ == "__main__":
    main()