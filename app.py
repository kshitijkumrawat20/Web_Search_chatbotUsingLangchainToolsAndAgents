import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper,BraveSearchWrapper
from langchain_community.utilities.golden_query import GoldenQueryAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from langchain_community.tools import YouTubeSearchTool, BraveSearch
from dotenv import load_dotenv
from langchain.agents import Tool


## Arxiv and wikipedia Tools
# brave_search = BraveSearch(search_wrapper=None)  # Pass search_wrapper=None
# brave_search_tool = Tool(
#     name="Brave-search",
#     description="Brave_Search",
#     func=brave_search.run
# 


youtube_search = YouTubeSearchTool()
youtube_search = YouTubeSearchTool()
youtube_search_tool = Tool(
    name="youtube-search",
    description="Search for YouTube videos. Input should be a simple search query string.",
    func=lambda query: youtube_search.run(query)
)
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)


api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")


st.title("SmartSearch Chatbot: Your Intelligent Web Companion! 🌐")

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

# intializing the chat history
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

# displaying the chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# handling the user input
if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    # setting up the llm
    llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant",streaming=True)
    tools=[arxiv,wiki,youtube_search_tool,search]

    # setting up the agent
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

    # setting up the callback
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        try:
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write(response)
        except ValueError as e:
            st.session_state.messages.append({'role': 'assistant', "content": f"Error: {str(e)}"})
            st.write(f"Error: {str(e)}")

