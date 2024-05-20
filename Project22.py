

import streamlit as st
from langchain import SQLDatabase, SQLDatabaseChain
from langchain.memory import ChatMessageHistory
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import QueryType
from azure.ai.formrecognizer import DocumentAnalysisClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate,ChatMessagePromptTemplate
from langchain.chat_models import ChatOpenAI    
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import time

from azure.search.documents.indexes.models import (  
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    PrioritizedFields,  
    SemanticField,  
    SearchField,  
    SemanticSettings,  
    VectorSearch,
    VectorSearchAlgorithmConfiguration,  
    CorsOptions,
    HnswParameters,
    
    
    
)




from PyPDF2 import PdfReader
import io
from fpdf import FPDF
import os
import openai


class proj():
    os.environ['OPENAI_API_KEY'] = "OpenAI key"
    openai.api_type = "azure" 
    openai.api_base = "https://cog-ppxxbrri2qmg6.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = "OpenAI key"
    
    

    def __init__(self):
        
        
        self.check=False

        self.uploaded_files = None

        self.endpoint = ""
        self.key = ""
        self.endpoint6 = ""
        self.key6 = ""

        # Create a SearchIndexClient instance
        self.credential = AzureKeyCredential(self.key)
        self.IndexName = "saved3"
        self.client2 = SearchClient(endpoint=self.endpoint, index_name=self.IndexName, credential=self.credential)
        self.client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        self.llm = OpenAI(temperature=1, openai_api_key="OpenAI Key", model_kwargs={"engine": "chat"})

        self.credential6 = AzureKeyCredential(self.key6)
        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=self.endpoint6, credential=AzureKeyCredential(self.key6)
        )

        
        self.Mainlist = []
        




        self.add_selectbox = st.sidebar.selectbox(
            "Is your question concerning the database or the document?",
            ('', 'SQL', 'PDF')
        )
        
        self.l=self.chekingIfTheirIsAnIndex(l=True)
        
       
        self.finalFunction()

    def chekingIfTheirIsAnIndex(self,l): 
        try:
            existing_index = self.client.get_index(self.IndexName)
        
        except Exception as e:
            if "No index" in str(e):
                l = False
            else:
                st.write(e)
        return l

    #"If l is False in the function 'chekingIfTheirIsAnIndex' Then this function will create an index in azure cognitive search"
    def IfLisFalse(self):
        
        index = SearchIndex(
            name=self.IndexName,
            fields = [
                SimpleField(name="Name", type=SearchFieldDataType.String, filterable=True,searchable=True, retrievable=True, key=True),
                SearchField(name="Document", type=SearchFieldDataType.String, filterable=True,searchable=True, retrievable=True),
                SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                                searchable=True, vector_search_dimensions=1536, vector_search_configuration="default"),
                        
                ],
                        
            semantic_settings=
                SemanticSettings(
                    configurations=[SemanticConfiguration(
                        name='default',
                        prioritized_fields=PrioritizedFields(
                            title_field=None, prioritized_content_fields=[SemanticField(field_name='Name')]))]),
                vector_search=VectorSearch(
                    algorithm_configurations=[
                    VectorSearchAlgorithmConfiguration(
                        name="default",
                        kind="hnsw",
                        hnsw_parameters=HnswParameters(metric="cosine"))
                                ]))

        cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
        try:
            result = self.client.create_index(index)
            st.write("An index have been created!")
        except Exception as e:
            st.write("An error have been detected ",e)



    #"This function Adds all the names of Documents that are already in azure cognitive search into a list called 'MainList' this is being done so that if the user wants to delete a documents he would choose the name of the document using the names in the MainList. The names in the index are keys"
    def NameOfPDFinMemory(self):
        search_query ="*"
        field_name ="Name"
        select_fields = [field_name,"Name"]
        results = self.client2.search(
            search_text=search_query,
            select=select_fields
        )
        count=0
                
        for result in results:
            field_value = result[field_name]
            self.Mainlist.append(field_value)
            check=True


    #When the user wants to ask a question 
    def AskaQuestion(self):
        


        solution=""
        styl = f"""
            <style>
                .stTextInput {{
                position: fixed;
                bottom: 3rem;
                }}
            </style>
            """
        st.markdown(styl, unsafe_allow_html=True)
        
        
        query = st.text_input("Please enter your PDF question:", value="",key="query_input",help="Type your message here...")
        #query=st.empty()
        #self.user_message(query)
        field_name = "Document"
        select_fields = [field_name,"Name"]

                
        count = 0
        string = []
                
        document1=""
        document2=""
        document3=""
        document4=""
        Name1=""
        Name2=""
        count7=0 
        
        
        
                        
        if (query != "" and query!=" "):

            originalQuery=query
            chat =ChatOpenAI(verbose=False,temperature=0,openai_api_key="Open AI Key", model_kwargs={"engine": "chat"})
            
            template = """
You will receive three documents and have access to the conversation history. Your task is to provide a solution based on the information in the documents without revealing the source document for your answers.

Instructions:
- If the documents do not contain a solution for the provided question, respond with an apology stating that you do not know the answer, and include the numbers 00112233 in the response.
- Please reply with an apology and 00112233 only when you do not know the answer.
- If a document contains the message "Now a table will be provided under this message!" it means the document has a table. The first line of the table represents the headers of the table. 
The information in the table will also be found in the text.
- Note that the headers in the table are in reverse order. For example, if the last header is "Name," then the first column in the table corresponds to names.
- Always respond in the same language as the given question.
- Use the conversation history if the question is related to the previous question or answer.
- If you want the user to clarify the answer, include 00112233.
- If you are not sure of your answer, send an apology while including 00112233.
- If there is not enough information, include 00112233.
- If the question is not clear, include in the response 00112233.
- If the information is not mentioned, include in the response 00112233.
- Always reply in Arabic.

Documents:
%Document1
{Document1}

%Document2
{Document2}

%Document3
{Document3}

Conversation History:
%History
{History}

YOUR RESPONSE:

"""
            template3 = """
                You will be given two questions, and your task is to determine if these two questions are related or not. Please respond with either "Yes" or "No."

                Instructions:
                - If you are unsure or do not have enough information to determine if the questions are related, respond with "Yes."
                - If the two questions are related, respond with "Yes."
                - If there is not enough information to determine whether they are related or not, reply with "Yes."

                For example:
                    question1: من هي السيدة: ريم محمد الردعان 
                    question2: و ما هو رقم المدني؟
                    Bot: Yes

                %Question1
                {Question1}

                %Question2
                {Question2}
        """
            

            
            template5 = """
                You will receive three documents. Your task is to provide a solution based on the information in the documents without revealing the source document for your answers.

                Instructions:
                - If the documents do not contain a solution for the provided question, respond with an apology stating that you do not know the answer, and include the numbers 00112233 in the response.
                - Please reply with an apology and 00112233 only when you do not know the answer.
                - If a document contains the message "Now a table will be provided under this message!" it means the document has a table. The first line of the table represents the headers of the table. The information in the table will also be found in the text.
                - Note that the headers in the table are in reverse order. For example, if the last header is "Name," then the first column in the table corresponds to names.
                - Use the conversation history if the question is related to the previous question or answer.
                - If you want the user to clarify the answer, include 00112233.
                - If you are not sure of your answer, send an apology while including 00112233.
                - If there is not enough information, include 00112233.
                - If the question is not clear, include in the response 00112233.
                - If the information is not mentioned, include in the response 00112233.
                - Always reply in Arabic.

                Documents:
                %Document1
                {Document1}

                %Document2
                {Document2}

                %Document3
                {Document3}

                

                YOUR RESPONSE:

            """

            
            
            template4 = """
                You will be given one question, and your task is to provide another way of asking the same question.

                Instructions:
                - If you did not understand the question, respond with "77777777881."
                - Do not attempt to provide an answer to the question.
                - Your response should only consist of another way to ask the same question.
                - Always respond in Arabic.

                %Question
                {Question}
"""




            prompt = PromptTemplate(input_variables=["Question1","Question2"], template=template3)
            

            
            #st.write(doc1)
            booleans=False
            if(len(st.session_state["chat_history"])!=0 and "00112233" not in st.session_state["chat_history"][-1]):
                final_prompt = prompt.format(Question1=st.session_state["chat_history"][-2].content,Question2=query)
                originalQuery=query
                response23 =chat(
                                [
                                    SystemMessage(content=final_prompt),
                                    HumanMessage(content="Check if Question1 and Question2 are related questions")
                                ]

                            )
                #st.write(response23.content," I am response number 23")
                if("Yes" in response23.content or "YES" in response23.content or "yes" in response23.content):
                    st.session_state["chat_history"].append(HumanMessage(content=query))
                    st.session_state["chat_history3"].append(HumanMessage(content=query, example=True))
                    count8=0
                    while(True):
                        
                        response =chat(
                                        
                                        st.session_state["chat_history"]
                                        
                                        
                                        

                                    )
                        if((count8!=5 and "00112233" in response.content) or(( "عذرًا"  in response.content or "توضيح السؤال" in response.content
                        or "توضيح" in response.content or "لا يوجد" in response.content or "للأسف" in response.content or "أعتذر" in response.content or "لم يتم ذكر" in response.content ) and count8!=5)):
                            prompt2 = PromptTemplate(input_variables=["Question"], template=template4)
                            final_prompt2 = prompt2.format(Question=query)
                            response23 =chat(
                                [
                                    SystemMessage(content=final_prompt2),
                                    HumanMessage(content="Ask the same question in a different way.")
                                ]

                            )
                            #st.write(response23.content)
                            query=response23.content
                            st.session_state["chat_history"][-1]=HumanMessage(content=query)
                            st.session_state["chat_history3"][-1]=HumanMessage(content=query, example=True)
                            count8+=1
                            continue.
                        else:
                            break
                    
                    if("00112233" not in response.content and "أعتذر" not in response.content and "عذرًا" not in response.content and "توضيح السؤال" not in response.content and "توضيح" not in response.content and "للأسف" not in response.content
                    and "لا يوجد" not in response.content and "لم يتم ذكر" not in response.content):
                        st.session_state["chat_history"][-1]=HumanMessage(content=originalQuery,example=True)
                        st.session_state["chat_history"].append(AIMessage(content=response.content))
                        st.session_state["chat_history3"][-1]=HumanMessage(content=originalQuery,example=True)
                        st.session_state["chat_history3"].append(AIMessage(content=response.content))
                        booleans=True
                    else:
                        query=originalQuery
                        st.session_state["chat_history"].pop(-1)



            if(booleans==False):
                response2=openai.Embedding.create(input=[query], engine="text-embedding-ada-002")
                    
                embedding_values2 = response2.data[0].embedding
                    
                results = self.client2.search(
                search_text=query,
                select=select_fields,
                top=2,
            )
                results2=self.client2.search(
                    search_text=query,
                    select=select_fields,
                    top=2,
                    vector=embedding_values2,
                    vector_fields="embedding"
                            

                )
                
                        
                    
                for result in results:
                    if(count7==0):
                        document1+=result[field_name]
                        Name1+=result["Name"]
                        count7+=1
                    else:
                        Name2+=result["Name"]
                        document2+=result[field_name]
                count7=0
                Name3=""
                Name4=""
                for result in results2:
                    if(count7==0):
                        document3+=result[field_name]
                        Name3+=result["Name"]
                        count7+=1
                    else:
                        Name4+=result["Name"]
                        document4+=result[field_name]
                prompt = PromptTemplate(input_variables=["Document1","Document2","Document3","History"], template=template)
                prompt5 = PromptTemplate(input_variables=["Document1","Document2","Document3"], template=template5)  

                    
                doc1=""
                doc2=""
                doc3=""
                if(Name1==Name3):
                    if(Name2==Name4):
                        
                        doc1=document1
                        doc2=document2
                        doc3=""
                    else:
                        
                        doc1=document1
                        doc2=document2
                        doc3=document4
                
                elif(Name1==Name4):
                    if(Name2==Name3):
                        
                        doc1=document1
                        doc2=document2
                        doc3=""
                    else:
                        
                        doc1=document1
                        doc2=document2
                        doc3=document3

                elif(Name2==Name4):
                    
                    doc1=document1
                    doc2=document2
                    doc3=document3
                    
                else:
                    if(Name2==Name3):
                        
                        doc1=document1
                        doc2=document2
                    else:
                        doc1=document1
                        doc2=document2
                        doc3=document3
                    
                chat_history2=[]
                for i in range(len(st.session_state["chat_history"])):
                    if(i!=0):
                        chat_history2.append(st.session_state["chat_history"][i])

                final_prompt = prompt.format(Document1=doc1,Document2=doc2,Document3=doc3,History=chat_history2)
                final_prompt5=prompt5.format(Document1=doc1,Document2=doc2,Document3=doc3)
                if(len(st.session_state["chat_history"])==0):
                    st.session_state["chat_history"].append(SystemMessage(content=final_prompt))
                else:
                    st.session_state["chat_history"][0]=SystemMessage(content=final_prompt)
                

                
                st.session_state["chat_history"].append(HumanMessage(content=query))
                st.session_state["chat_history3"].append(HumanMessage(content=query, example=True))

                count88=0
                count8=0
                count9=0
                checking=False
                while(True):
                    try:
                        response =chat(
                                        
                                        st.session_state["chat_history"]
                                        
                                        
                                        

                                    )
                        if(count88==0 and ("00112233" in response.content or "عذرًا"  in response.content or "أعتذر" in response.content or "توضيح السؤال" in response.content
                        or "توضيح" in response.content or "لا يوجد" in response.content or "للأسف" in response.content or "لم يتم ذكر" in response.content)):
                            count88+=1
                            response =chat(
                                SystemMessage(content=final_prompt5),
                                HumanMessage(content=originalQuery)
        
                            )

                        if(count8!=5 and "00112233" in response.content or(( "عذرًا"  in response.content or "أعتذر" in response.content or "توضيح السؤال" in response.content
                        or "توضيح" in response.content or "لا يوجد" in response.content or "للأسف" in response.content or "لم يتم ذكر" in response.content) and count8!=5)):
                            prompt2 = PromptTemplate(input_variables=["Question"], template=template4)
                            final_prompt2 = prompt2.format(Question=query)
                            response23 =chat(
                                [
                                    SystemMessage(content=final_prompt2),
                                    HumanMessage(content="Ask the same question in a different way.")
                                ]

                            )
                            #st.write(response23.content)
                            query=response23.content
                            st.session_state["chat_history"][-1]=HumanMessage(content=query)
                            st.session_state["chat_history3"][-1]=HumanMessage(content=query, example=True)
                            count8+=1
                            continue
                        else:
                            break
                    except:
                        if(len(st.session_state["chat_history"])>2):
                            #st.write("An Error have been detected!! please wait few moments!!!")
                            st.session_state["chat_history"].pop(1)
                            st.session_state["chat_history"].pop(1)
                            

                           
                        else:
                            
                            
                            if(count9==0):
                                final_prompt = prompt.format(Document1=doc1,Document2=doc2,Document3="",History=chat_history2)
                                st.session_state["chat_history"][0]=SystemMessage(content=final_prompt)
                                count9+=1
                            elif(count9==1):
                                final_prompt = prompt.format(Document1=doc1,Document2="",Document3="",History=chat_history2)
                                st.session_state["chat_history"][0]=SystemMessage(content=final_prompt)
                                count9+=1
                            else:
                                #st.write("أنا لآ أعرف الجواب")
                                checking=True
                                break
                    
                    
                    
                
                if(checking==False):
                    st.session_state["chat_history"][-1]=HumanMessage(content=originalQuery,example=True)
                    st.session_state["chat_history"].append(AIMessage(content=response.content))
                    st.session_state["chat_history3"][-1]=HumanMessage(content=originalQuery,example=True)
                    st.session_state["chat_history3"].append(AIMessage(content=response.content))
                    query=""
        for i in range(len(st.session_state["chat_history3"])):
            
            l33=st.session_state["chat_history3"][i]
            
            if(l33.example==True):
                st.write(" ")
                self.user_message(st.session_state["chat_history3"][i])
                st.write(" ")
            else:
                st.write(" ")
                self.bot_message(st.session_state["chat_history3"][i])
                st.write(" ")
            
        if(len(st.session_state["chat_history"])==st.session_state["counting"]):
            st.markdown("""
                <style>
                .scroll-down {
                    animation: scroll-down 10s infinite;
                }

                @keyframes scroll-down {
                    0%, 100% {
                        transform: translateY(0);
                    }
                    50% {
                        transform: translateY(100vh);
                    }
                }
                </style>
                <div class="scroll-down"></div>
            """, unsafe_allow_html=True)
            st.session_state["counting"]=2+st.session_state["counting"]
                

    def user_message(self, message):
        st.markdown(
            f'<div style="text-align: right;">'
            f'<span style="background-color: #DCF8C6; padding: 5px; border-radius: 10px;">'
            f'You: {message.content}'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Define a function to display bot messages on the left
    def bot_message(self,message):
        st.markdown(
            f'<div style="text-align: left;">'
            f'<span style="background-color: #EAEAEA; padding: 5px; border-radius: 10px;">'
            f'Bot: {message.content}'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True
        )



    #"This method is creating the table. In the AddMore method when i reach i paragraph that has the same location as word in a table then i assumed that this word is in a table not a paragraph therfore if this word in a table is the last word in the colum new line else \t\t|"
    def Tables(self,l3,strrr,counts):
        l22=l3.cells
        k=""

                
        if(l22[counts].column_index==(l3.column_count-1)):
            strrr+=l22[counts].content+"""

    """
        elif(l22[counts].column_index<(l3.column_count-1)):
            strrr+=l22[counts].content+"    |    "
        return strrr
       

    #To Delete an index in azure cognitive search However unfortinatlly the sentence 
    def Delete(self,delete_name):
        delete_name=st.sidebar.selectbox("Which pdf whould you like to delete",(' ',) + tuple(self.Mainlist))
        check=st.selectbox("Are you sure you want to delete this pdf?",(' ','Yes','No',))
        if(delete_name!=" " and delete_name!="l"):
            if(check=='Yes'):
                boolean=True
                newList=[]
                for i in range(len(self.Mainlist)):
                    if(self.Mainlist[i]!=delete_name):
                        newList.append(self.Mainlist[i])
                for i in range(len(newList)):
                    newList[i]=self.Mainlist[i]
                    
                document_key=delete_name
                self.client2.delete_documents(documents=[{"Name":delete_name}])
                delete_name="l"
                st.experimental_rerun()

    
    
    
    def AddMore(self,upload_files):
        uploaded_bytes = self.uploaded_files.read()
        pdf_reader = PdfReader(io.BytesIO(uploaded_bytes))
        num_pages = len(pdf_reader.pages)
        pdf_bytes = self.uploaded_files.read()
        strrr2 = ""
        c = 0
        for l in self.uploaded_files.name:
            if l == '.':
                break
            if l == '.' or l == ' ':
                c = c + 1
            else:
                strrr2 += l
        count2 = 1
        paragraph = ""
        count3,count4,count5,count6,count7=0
        booll=True
        if(pdf_bytes!=None):
            uploaded_file_object = self.uploaded_files.getvalue()
            poller = self.document_analysis_client.begin_analyze_document("prebuilt-layout", document=uploaded_file_object)
            result = poller.result()
            l=result.paragraphs
            counts=0
            try:
                l3=result.tables[0]
                l2=result.tables[0].cells[counts].bounding_regions[0].polygon
            except:
                l3=None
            docu=""
            strrr=""
            check=False
            Stopp=False
            if(l3 ==None):
                for parag in l:
                    docu+=parag.content+"""

            """
            else:   
                for prag in l:
                    
                    if(prag.bounding_regions[0].polygon == l2 and Stopp==False): 
                        strrr=self.Tables(l3,strrr,counts)
                        counts=counts+1
                        check=True
                        try:
                            l2=result.tables[0].cells[counts].bounding_regions[0].polygon
                        except:
                            Stopp=True
                        
                        
                    else:
                        if(check==True):
                            if(Stopp==False):
                                if(l3.cells[counts].row_index!=0):
                                    strrr=self.Tables(l3,strrr,counts)
                                    counts=counts+1
                                    check=True
                                    try:
                                        l2=result.tables[0].cells[counts].bounding_regions[0].polygon
                                    except:
                                        Stopp=True
                                

                            else:
                                docu+="""

            """
                                docu+="Now a table will be provided under this message!"
                                docu+="""
                                        
            """
                                        
                                docu+=strrr
                                strrr=""
                                check=False

                        else:
                            docu+=prag.content+"""

            """

            response=openai.Embedding.create(input=[docu], engine="text-embedding-ada-002")
            embedding_values = response.data[0].embedding

            document = {
                            "Name": f"{strrr2}",
                            "Document": docu,
                            "embedding":embedding_values
                                    
                        }
        st.write(docu)
        response =self.client2.upload_documents(documents=[document])
        if response[0].succeeded:
            st.write("Document uploaded successfully.")
        else :
            st.write("An Error occuerd when uploading the document: ")





    def finalFunction(self):

        if self.add_selectbox != "":
            st.sidebar.write("You choose ", self.add_selectbox)
            if self.add_selectbox == "PDF":

                if (self.l==False):
                    self.IfLisFalse()


                

                add_selectbox2 = st.sidebar.selectbox(
                    "In which topic do you want to ask questions?",
                    (' ', 'Ask a question', 'Add More', 'Delete',)
                )
                if(len(self.Mainlist)==0 and self.check==False):
                    self.NameOfPDFinMemory()
                    
                if(add_selectbox2 =='Ask a question'):
                   self.AskaQuestion()
                   



                elif(add_selectbox2=='Delete'):
                    self.Delete(delete_name=" ")
                        
                        
                        
                        

            
                elif(add_selectbox2=='Add More'):
                    st.write()
                    self.uploaded_files = st.file_uploader("Please Upload a pdf file", type=["pdf"], accept_multiple_files=False)

                if self.uploaded_files is not None and self.uploaded_files != "":
                    self.AddMore(upload_files=self.uploaded_files)
            elif(self.add_selectbox=="SQL"):
                mysql_username = "root"
                mysql_password = ""
                mysql_host = "localhost"
                mysql_port = "3306"
                mysql_database_name = "test2"
                mysql_uri = f"mysql+mysqlconnector://{mysql_username}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_database_name}"

                db = SQLDatabase.from_uri(mysql_uri)

                db_chain = SQLDatabaseChain(llm=self.llm, database=db, verbose=True)



                # Prompt Template


                # User Question
                
                question = st.text_input("Please enter your SQL question:")
                if(question!="" and question!=" "):

                    
                # Execute the Database Chain
                    try:

                        response = db_chain.run(question)

                        st.write("Response from Language Model:")
                        strr=""
                        for i in response:
                            if(i!='\n'):
                                strr+=i
                            else:
                                break
                        st.write(strr)
                        
                        




                    except Exception as e:
                        print()
                        st.write("An Error has beed detected!!!!")
                        print()
                        print()
                        st.write(e)













if "chat_history" not in st.session_state:
    
    st.session_state["chat_history"]=[]

#I am not using chat_history2 
if("chat_history2" not in st.session_state):
    st.session_state["chat_history2"]=[]
if("counting" not in st.session_state):
    st.session_state["counting"]=5

if("chat_history3" not in st.session_state):
    st.session_state["chat_history3"]=[]


page_be_image="""
<style>
[data-testid="stAppViewContainer"]{
background-color: #E8E8B5;
}
</style>
"""

st.markdown(page_be_image,unsafe_allow_html=True)


page_be_image2="""
<style>
[data-testid="stHeader"]{
background-color: #E8E8B5;
}
</style>
"""
st.markdown(page_be_image2,unsafe_allow_html=True)



problem=proj()








                
                
                

                            


                        
                



