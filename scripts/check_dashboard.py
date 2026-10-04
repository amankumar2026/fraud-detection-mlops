"""Runs dashboard/app.py through Streamlit's test harness and reports any
exceptions, so we know the page actually renders instead of just being served.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
at = AppTest.from_file(app_path, default_timeout=300).run()

if at.exception:
    print("EXCEPTIONS:")
    for e in at.exception:
        print(e.value)
else:
    print("No exceptions. Metrics shown:")
    for m in at.metric:
        print(f"  {m.label}: {m.value}")
