from pathlib import Path
import streamlit as st

base_dir = Path(__file__).resolve().parent
pages_dir = base_dir / "pages"

Page1 = st.Page(str(pages_dir / "page1.py"), title="Home")
Page2 = st.Page(str(pages_dir / "page2.py"), title="Fake News Detection Tool")
Page3 = st.Page(str(pages_dir / "page3.py"), title="Model Analysis")

pg = st.navigation({"News: Real or Fake": [Page1, Page2, Page3]})
pg.run()
