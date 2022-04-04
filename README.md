[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/fzj-iek3-vsa/nostream)


<a href="https://www.fz-juelich.de/iek/iek-3/DE/Home/home_node.html"><img src="https://www.fz-juelich.de/metis-platform/EN/_Documents/Pictures/FZJ-logo_340x185.jpg?__blob=normal
" alt="Forschungszentrum Juelich Logo" width="230px"></a> 


# Wie sicher ist die Energieversorgung ohne russisches Erdgas?

Die europäische Energieversorgung ist in hohem Maße auf Energieimporte angewiesen. Dies gilt in besonderem Maße auch für Deutschland: Fast drei Viertel des deutschen
Energieverbrauchs wird aktuell importiert. Insbesondere bei Erdgas ist die Importabhängigkeit mit einer Quote von ca. 94% besonders ausgeprägt. Mit einem Anteil von über 50% dominiert Russland die derzeitigen Erdgasimporte nach Deutschland.

![image](https://user-images.githubusercontent.com/63047357/159122405-15384030-474a-4b83-9962-f127e15a9006.png)


## Setup and develop

In order to locally develop the application, install anaconda and run

```bash
conda env create -f environment.yml
```

afterwards activate the environment, move into the streamlit repository and start streamlit with

```bash
streamlit run ./streamlit/streamlit_app.py
```



## Change log


#### 0.2 - 04.04.2022

* Add the possibility to add additional pipeline imports, e.g., from Norway
* Reduce also LNG imports from Russia with a starting embargo
* Change the merit order of curtailemt, with the industry first and export last


#### 0.1 - 30.03.2022

* First publice release