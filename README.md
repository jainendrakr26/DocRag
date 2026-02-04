# DocRag
We are using uv in place of pip as it is fast.(https://docs.astral.sh/uv/)
#initialize uv
uv init

#create virtual env
uv venv

#Activate virtual env
source .venv/bin/activate

#Add all the things that are needed in the project
uv add -r requirements.txt

#Add kernel
uv add ipykernel

#create .env file and add you OPENAI_API_KEY

To run the project,
streamlit run streamlit_app.py