import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

import io
from pathlib import Path
from matplotlib.colors import to_hex
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


import pytesseract


# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Menu", ["Home",'Palette','ImageToText', 'Uber - Streamlit tutorial', 'Sobre mim'], 
        icons=['house','','', 'geo-alt', 'gear'], menu_icon="cast", default_index=0)

    
if selected == "Home" :
    st.title('home')
#############################################################################################################################################
if selected == "Palette" :
    def get(imagem_carregada, n_cores):
        # salvar a imagem do streamlit
        with open('imagem.jpg', 'wb') as file:
            file.write(imagem_carregada.getbuffer())
        # ler a imagem
        image = Image.open('imagem.jpg')
        # transformar os pixels em linhas de uma matriz
        N, M = image.size
        X = np.asarray(image).reshape((M*N, 3))
        # criar e aplicar o k-means na imagem
        model = KMeans(n_clusters=n_cores, random_state=42).fit(X)
        # capturar os centros (cores médias dos grupos)
        cores = model.cluster_centers_.astype('uint8')[np.newaxis]
        cores_hex = [to_hex(cor/255) for cor in cores[0]]

        # apagar imagem salva
        Path('imagem.jpg').unlink()
        # retornar cores
        return cores, cores_hex

    def show(cores):
        fig = plt.figure()
        plt.imshow(cores)
        plt.axis('off')
        return fig

    def save(fig):
        img = io.BytesIO()
        fig.savefig(img, format='png')
        plt.axis('off')
        return img

    st.title("Gerador de paletas")
    imagem = st.file_uploader("Envie sua imagem", ["jpg", "jpeg"])

    col1, col2 = st.columns([.7, .3])

    if imagem:
        col1.image(imagem)
        n_cores = col2.slider(
            "Quantidade de cores",
            min_value=2,
            max_value=8,
            value=5
        )
        botao_gerar_paleta = col2.button("Gerar paleta!")
        if botao_gerar_paleta:
            cores, cores_hex = get(imagem, n_cores)
            figura = show(cores)
            col2.pyplot(fig=figura)
            col2.code(f"{cores_hex}")

            col2.download_button(
                "Download",
                save(figura),
                "paleta.png",
                'image/png'
            )
            
#############################################################################################################################################
if selected == "ImageToText" :
    st.title('Image to text')
    imgs = st.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    for img in imgs:
        image = Image.open(img)
        st.image(image, caption='Imagem carregada', use_column_width=True)
        #pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        #txt = pytesseract.image_to_string(image, lang='por')
        #pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
        txt = pytesseract.image_to_string(image)
        st.write(txt)
                

#############################################################################################################################################
if selected == "Uber - Streamlit tutorial" :
    st.title('Uber pickups in NYC')


    DATE_COLUMN = 'date/time'
    DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')


    @st.cache
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data


    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.

    data = load_data(1000)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

    number_of_pickups_by_hour = st.checkbox('Show Number of pickups by hour')


    if number_of_pickups_by_hour:
        st.subheader('Number of pickups by hour')
        hist_values = np.histogram(
            data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

        st.bar_chart(hist_values)    


    hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h

    st.subheader('Map of all pickups')

    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
    st.subheader(f'Map of all pickups at {hour_to_filter}:00')
    st.map(filtered_data)

    
if selected == "Sobre mim" :
    st.title('Sobre mim')
    
