import streamlit as st
import requests
import torchaudio
import io

st.title('URBAN Audio')

uploaded_file = st.file_uploader('Выберите аудиофайл', type=['wav'])

if uploaded_file is not None:

    st.audio(uploaded_file)


    if st.button('check'):
        url_backend = 'http://127.0.0.1:8000/predict/'

        uploaded_file.seek(0)

        files = {'file':uploaded_file}

        try:
            answer = requests.post(url=url_backend, files=files)
            if answer.status_code == 200:
                st.json(answer.json())
            else:
                st.error(f'Ошибка бэкенда: {answer.status_code}')
        except requests.exceptions.RequestException as e:
            st.error(f'Не удалось связаться с сервером: {e}')


