from systemy_bliczeniowe.lab1.listaP import download_files

URLS = [
    'https://unsplash.com/photos/Q2BolmJ7E9U',
    'https://unsplash.com/photos/Im2YhZG-ccQ',
]
DIR = 'lab1_data'

download_files(URLS, DIR)
