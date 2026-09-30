"""Dashboard entry point: `streamlit run streamlit_app.py`.

Lives at the repository root because Streamlit puts the launched script's
own folder on the import path: from here that's the root, so `import src...`
resolves; launching src/dashboard.py directly would put src/ there instead.
It's also the filename Streamlit Community Cloud looks for by default.
"""

from src.dashboard import main

main()
