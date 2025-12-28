import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import re
from nlp.debris_knowledge_graph import DebrisKnowledgeGraph

class OrbitGPTEngine:
    def __init__(self, cdm_service):
        self.cdm_service = cdm_service
        self.persist_directory = "./chroma_db"
        
        # Initialize Embedding Model (Local, fast)
        self.embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize Vector Store
        self.vector_db = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embedding_fn
        )
        
        # Initialize LLM (Ollama - ensure 'llama3' or 'mistral' is pulled)
        # fallback to a default if not found
        self.llm = Ollama(model="llama3")
        
        # Initialize Knowledge Graph
        self.kg = DebrisKnowledgeGraph()

    def ingest_cdms(self, sat_id=25544):
        """Fetch CDMs and store them in the Vector DB."""
        print("Fetching fresh CDMs...")
        cdms = self.cdm_service.fetch_recent_cdms(sat_id)
        if not cdms:
            return "No CDMs found to ingest."

        texts = []
        metadatas = []
        ids = []
        
        for cdm in cdms:
            # Create natural language summary
            text = self.cdm_service.parse_cdm_to_text(cdm)
            cdm_id = cdm.get("CDM_ID", "unknown")
            
            texts.append(text)
            metadatas.append({"source": "space-track", "cdm_id": cdm_id})
            ids.append(cdm_id)
            
        # Add to Chroma
        self.vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"Ingested {len(texts)} CDMs into OrbitGPT Knowledge Base.")
        return f"Ingested {len(texts)} CDMs."

    def ask(self, query):
        """Ask a question to OrbitGPT."""
        
        # 0. Query Knowledge Graph (Extract Entitites)
        graph_context = ""
        # Look for 5-digit NORAD IDs
        ids = re.findall(r'\b\d{5}\b', query)
        if ids:
            print(f"OrbitGPT found IDs: {ids}")
            for nid in ids:
                attr = self.kg.query_attribution(nid)
                if attr:
                     graph_context += f"Attrib({nid}): Owner={attr['owner']}, Parent={attr['parent']}, Event={attr['origin_event']}\n"
        
        if not graph_context and "fengyun" in query.lower():
             # Fallback context injection for non-ID queries
             graph_context += "Attrib(Fengyun): Owner=China, Parent=None, Event=2007-ASAT-Test\n"

        # Create RAG Chain
        prompt_template = f"""
        You are OrbitGPT, an expert Flight Dynamics & Space Law Assistant.
        
        Knowledge Graph Context (Attribution):
        {graph_context}
        
        Document Context (CDMs):
        {{context}}
        
        User: {{question}}
        
        Instructions:
        1. Use the Knowledge Graph to identify WHO owns the object and WHERE it came from.
        2. Use the CDMs to identify the Risk level (TCA, Probability).
        3. Combine both to give a "Space Lawyer" recommendation.
        
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        try:
            # Check if DB has data
            if self.vector_db._collection.count() == 0:
                print("Vector DB empty. Ingesting global fallback...")
                self.ingest_cdms(sat_id=25544) # Try Default

            result = qa_chain.invoke({"query": query})
            return result["result"]
        except Exception as e:
            print(f"OrbitGPT Query Error: {e}")
            return "I am currently unable to access the risk database. (Is Ollama running locally?)"

if __name__ == "__main__":
    from cdm_service import CDMService
    svc = CDMService()
    bot = OrbitGPTEngine(svc)
    
    # 1. Ingest
    bot.ingest_cdms()
    
    # 2. Ask
    print("--- User: Are there any high risk conjunctions? ---")
    response = bot.ask("Are there any high risk conjunctions?")
    print("OrbitGPT:", response)
