[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://no-stream.fz-juelich.de/)


[![Forschungszentrum Juelich Logo](./static/FJZ_IEK-3.svg)](https://www.fz-juelich.de/iek/iek-3/DE/Home/home_node.html)

# Wie sicher ist die Energieversorgung ohne russisches Erdgas?
Mit dem Krieg in der Ukraine steht ein Stopp der Erdgaslieferungen aus Russland im Raum. Eine neue Web-Applikation (https://no-stream.fz-juelich.de/) des Forschungszentrums Jülich macht es nun möglich, die Folgen eines solchen – kompletten oder teilweisen – Embargos auf die deutschen Erdgasvorräte zu ermitteln.

[![NoStream app](./static/NoStream_interface.PNG)](https://no-stream.fz-juelich.de/)


## Setup and develop

In order to locally develop the application, install anaconda and run

```bash
pip install -r requirements.txt
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
