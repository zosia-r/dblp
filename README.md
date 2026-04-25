# dblp

## STRATEGIA
1. ETL - done
2. TOPIC DISCOVERY
    - TF-IDF na tytułach
    - clustering: K-means albo BERTopic
3. EDA
    - liczba publikacji na rok
    - top autorzy
    - średnia liczba autorów na papier
    - najczęstsze słowa w tytułach
    - graf współautorstwa
4. SYSTEM REKOMENDACJI
    - podobieństwa między papierami (TF-IDF + cosine similarity)
    - embeddings
5. SEARCH / CHATBOT
    - prosty interfejs do wyszukiwania publikacji
    - proste zapytania o trendy (np. "jakie są najpopularniejsze tematy w 2020 roku?")
5. DASHBOARD
    - Streamlit

16:42:43 [INFO] XML parse done 
| total: 12_501_682 
| article: 4_249_812 
| inproceedings: 3_860_628 
| proceedings: 63_924 
| book: 21_374 
| incollection: 71_063 
| phdthesis: 152_212 
| mastersthesis: 27 
| www: 4_059_905 
| person: 0 
| data: 22_737

## wybór danych
- article - An article from a journal or magazine - 34% wszystkich rekordów
- inproceedings - A paper in a conference or workshop proceedings - 31% wszystkich rekordów
- phdthesis - A PhD thesis - 1% wszystkich rekordów
- www - person records - 32% wszystkich rekordów
    - allows to link papers to authors
    - allows person disambiguation



## Dane do analizy
1. Paper
    - title
    - year
    - venue
    - type
    - authors
        - name
        - aliases
