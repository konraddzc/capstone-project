import streamlit as st

Home = st.Page("pages/page1.py", title="Home")

pg = st.navigation({"News: Real or Fake": [Home]})

pg.run()
