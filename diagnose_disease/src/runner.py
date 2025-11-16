from diagnose.context import DiagnoseContext
from diagnose.diagnose import Diagnose
import os
import pandas as pd

if __name__ == "__main__":
    ctx = DiagnoseContext(
        llm_provider=os.environ.get("LLM_PROVIDER"),
        llm_provider_model=os.environ.get("LLM_PROVIDER_MODEL"),
        llm_provider_key=os.environ.get("LLM_PROVIDER_KEY"),
    )

    diagnose = Diagnose(ctx)
    diagnose.run()
