"""Reference: https://github.com/teddylee777/langchain-teddynote/blob/main/langchain_teddynote/logging.py"""

import os


def langsmith(project_name=None, set_enable=True):

    if set_enable:
        langchain_key = os.environ.get("LANGCHAIN_API_KEY", "")
        langsmith_key = os.environ.get("LANGSMITH_API_KEY", "")

        # 더 긴 API 키 선택
        if len(langchain_key.strip()) >= len(langsmith_key.strip()):
            result = langchain_key
        else:
            result = langsmith_key

        if result.strip() == "":
            print(
                "LangChain/LangSmith API Key is not set. Reference: https://wikidocs.net/250954"
            )
            return

        os.environ["LANGSMITH_ENDPOINT"] = (
            "https://api.smith.langchain.com"  # LangSmith API endpoint
        )
        os.environ["LANGSMITH_TRACING"] = "true"  # true: enabled
        os.environ["LANGSMITH_PROJECT"] = project_name  # project name
        print(f"Enabled LangSmith Tracing.\n[Project Name]\n{project_name}")
    else:
        os.environ["LANGSMITH_TRACING"] = "false"  # false: disabled
        print("LangSmith Tracing is disabled.")


def env_variable(key, value):
    os.environ[key] = value