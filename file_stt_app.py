# Libraries to be used ------------------------------------------------------------

import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speech_sdk
# Add OpenAI library
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

def speech_recognize_continuous_from_file(filename):
    import time

    # """performs continuous speech recognition with input from an audio file"""
    # <SpeechContinuousRecognitionWithFile>
    audio_config = speech_sdk.AudioConfig(filename=filename)

    speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_config)

    done = False
    all_results = []
    text_transcribing = "Procesando..."

    def stop_cb(evt: speech_sdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING')
        nonlocal done
        done = True
    
    def speech_recognizer_recognition_canceled_cb(evt: speech_sdk.SessionEventArgs):
        print('Canceled event')

    def speech_recognizer_session_stopped_cb(evt: speech_sdk.SessionEventArgs):
        print('SessionStopped event')

    def speech_recognizer_recognizing_cb(evt: speech_sdk.SpeechRecognitionEventArgs):
        nonlocal text_transcribing
        text_transcribing = evt.result.text
        print('Transcribing: ', text_transcribing)

    def speech_recognizer_transcribed_cb(evt: speech_sdk.SpeechRecognitionEventArgs):
        # global sitio
        # sitio = st.empty()
        print('TRANSCRIBED:')
        if evt.result.reason == speech_sdk.ResultReason.RecognizedSpeech:
            print(f'\tText: {evt.result.text}')
            all_results.append(evt.result.text)
        elif evt.result.reason == speech_sdk.ResultReason.NoMatch:
            print(f'\tNOMATCH: Speech could not be TRANSCRIBED: {evt.result.no_match_details}')
            stop_cb(evt)

    def speech_recognizer_session_started_cb(evt: speech_sdk.SessionEventArgs):
        print('SessionStarted event')

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(speech_recognizer_recognizing_cb)
    speech_recognizer.recognized.connect(speech_recognizer_transcribed_cb)
    speech_recognizer.session_started.connect(speech_recognizer_session_started_cb)
    speech_recognizer.session_stopped.connect(speech_recognizer_session_stopped_cb)
    speech_recognizer.canceled.connect(speech_recognizer_recognition_canceled_cb)
    # stop transcribing on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)
        with sitio:
            print('T: [' + text_transcribing + ']')
            st.info(text_transcribing + '\n')
            
            #texto_final = ""
            #for text in all_results:
            #    texto_final += text + " \n"
            #st.info(texto_final)

    speech_recognizer.stop_continuous_recognition()

    return all_results

def openai_functions(text):
    try:
        load_dotenv()
        my_api_key=os.getenv("AZURE_OPENAI_API_KEY")
        my_azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

        client = AzureOpenAI(
            api_key=my_api_key,
            api_version="2023-12-01-preview", #2023-03-15-preview
            azure_endpoint =my_azure_endpoint
        )

        message_text = [
                {"role": "system", "content": "Eres un agente experto en hacer resumenes de conversaciones, identificar entidades y clasificar el sentimiento en positivo, negativo o neutro."},
                {"role": "user", "content": "Resume la siguiente conversación de una llamada a un contact center marcado entre triple comillas, extrae las entidades clave e identifica el sentimiento general '''" + text + "'''"}
            ]

        response = client.chat.completions.create(
            model=deployment_name,
            messages = message_text
        )

        return response.choices[0].message.content

    except Exception as ex:
        print(ex)

def openai_functions_sk(text):
    try:
        import semantic_kernel as sk
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion

        kernel = sk.Kernel()

        # Prepare Azure OpenAI service using credentials stored in the `.env` file
        deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
        print(f"deployment: [{deployment}], api_key: [{api_key}], endpoint: [{endpoint}]")
        kernel.add_chat_service("dv", AzureChatCompletion(deployment_name=deployment, base_url=endpoint, api_key=api_key))

        # Wrap your prompt in a function
        text_prompt = f"""
                Eres un agente experto en hacer resumenes de conversaciones, 
                identificar entidades y clasificar el sentimiento en positivo, negativo o neutro.
                Resume la siguiente conversación de una llamada a un contact center marcado entre triple comillas,
                extrae las entidades clave e identifica el sentimiento general '''{text}'''
            """
        print(f'prompt: [{text_prompt}]')

        prompt = kernel.create_semantic_function(text_prompt)

        # Run your prompt
        print(prompt())

    except Exception as ex:
        print(ex)

# from css_tricks import _max_width_

# title and favicon ------------------------------------------------------------

st.set_page_config(page_title="Azure Speech to Text Demo", page_icon="👄")

# _max_width_()

# logo and header -------------------------------------------------

st.text("")
st.image("microsoft.png", width=125)
st.title("Azure Speech to text demo")
st.write("""Upload a wav file, transcribe it, then export it to a text file""")

#st.text("")

c1, c2, c3 = st.columns([1, 4, 1])

with c2:
    f = st.file_uploader("", type=[".wav"])
        #st.info(f"""Upload a .wav file. Try a sample: [Sample 01](https://github.com/CharlyWargnier/CSVHub/blob/main/Wave_files_demos/Welcome.wav?raw=true) | [Sample 02](https://github.com/CharlyWargnier/CSVHub/blob/main/Wave_files_demos/The_National_Park.wav?raw=true)""")
    button = st.button(label="Transcribe")

if button:
    st.audio(f, format="wav")
    path_in = f.name

    sitio = st.empty()
    with sitio:
        st.info("Procesando...")

    try:
        # Get Configuration Settings
        load_dotenv()
        speech_key = os.getenv('SPEECH_KEY')
        speech_region = os.getenv('SPEECH_REGION')

        # Configure speech service
        language = "es-es"
        speech_config = speech_sdk.SpeechConfig(subscription=speech_key, region=speech_region, speech_recognition_language=language)
        print(f'Ready to use speech service in region {speech_config.region} and in language {language}')
        
        # Transcribe file
        current_dir = os.getcwd()
        audioFile = current_dir + '\\' + path_in
        # Configure speech recognition
        audio_config = speech_sdk.AudioConfig(filename=audioFile)
        speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_config)

        texts=speech_recognize_continuous_from_file(path_in)
        texto_final = ""
        for text in texts:
            texto_final += text + "\n"
        print(texto_final)

        st.write("Análisis de la conversación:")
        response=openai_functions(texto_final)
        #response=openai_functions_sk(texto_final)
        st.info(response)
        
    except Exception as ex:
        print(ex)

    # st.info(texto_final)

    c0, c1 = st.columns([2, 2])

    with c0:
        st.download_button(
            "Download the transcription",
            texto_final,
            file_name=None,
            mime=None,
            key=None,
            help=None,
            on_click=None,
            args=None,
            kwargs=None,
        )


else:
    path_in = None
    st.stop()
