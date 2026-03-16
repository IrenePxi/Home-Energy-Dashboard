from __future__ import annotations

import streamlit as st

from services.paths import rpath


def render_data_guide(md_rel_path: str = "HOME_ENERGY_DASHBOARD_AI_SPEC.md") -> None:
    with st.container(border=True):
        st.markdown("#### 📄 Dashboard Data Guide")

        md_path = rpath(md_rel_path)
        try:
            content = md_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            st.warning(f"Guide file not found: `{md_rel_path}`")
            return
        except Exception as e:
            st.error(f"Could not read guide file: {e}")
            return

        with st.expander("Open guide", expanded=False):
            st.markdown(content)

