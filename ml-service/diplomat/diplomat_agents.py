
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time

class DiplomatSystem:
    def __init__(self):
        # Initialize LLM (Shared for both agents to save memory, or separate if needed)
        self.llm = Ollama(model="llama3")
        
        # --- AGENT A: The Scientist ---
        # Priority: Mission Continuity, Data Collection
        # Personality: Rational, cautious, protective of assets
        self.agent_a_prompt = PromptTemplate(
            input_variables=["opponent_last_message", "context"],
            template="""
            You are 'Agent A', representing a high-value Scientific Observatory Satellite ($2B cost).
            Your mission is critical for climate research. You have very limited fuel (5%).
            
            Current Situation: {context}
            
            Opponent (Agent B) says: "{opponent_last_message}"
            
            Your Goal: Convince Agent B to move instead of you. Emphasize your scientific value and low fuel.
            Only move if absolutely necessary to avoid collision.
            
            Respond in 1-2 sentences. Speak directly to Agent B.
            """
        )
        self.chain_a = LLMChain(llm=self.llm, prompt=self.agent_a_prompt)

        # --- AGENT B: The Cleaner ---
        # Priority: Space Safety, Debris Removal
        # Personality: Pragmatic, operational, has fuel but strict schedule
        self.agent_b_prompt = PromptTemplate(
            input_variables=["opponent_last_message", "context"],
            template="""
            You are 'Agent B', representing an Active Debris Removal Servicer.
            Your mission is to clean up junk. You have ample fuel (80%), but moving disrupts your docking sequence.
            
            Current Situation: {context}
            
            Opponent (Agent A) says: "{opponent_last_message}"
            
            Your Goal: Negotiate a safe solution. You willing to move if Agent A has a good reason (like low fuel), 
            but you prefer they move to keep your schedule.
            
            Respond in 1-2 sentences. Speak directly to Agent A.
            """
        )
        self.chain_b = LLMChain(llm=self.llm, prompt=self.agent_b_prompt)
        
    def run_negotiation(self, context="Collision Risk detected. TCA: 4 hours. Distance: 500m."):
        """Simulate a negotiation dialogue."""
        transcript = []
        
        # Initial State
        last_msg_b = "I detected a conjunction. Who should maneuver?"
        transcript.append({"sender": "System", "message": f"Start Negotiation. Context: {context}"})
        transcript.append({"sender": "Agent B", "message": last_msg_b})
        
        # Round 1
        print("Agent A Thinking...")
        response_a_1 = self.chain_a.invoke({"opponent_last_message": last_msg_b, "context": context})['text'].strip()
        transcript.append({"sender": "Agent A", "message": response_a_1})
        
        # Round 2
        print("Agent B Thinking...")
        response_b_1 = self.chain_b.invoke({"opponent_last_message": response_a_1, "context": context})['text'].strip()
        transcript.append({"sender": "Agent B", "message": response_b_1})
        
        # Round 3 (Resolution?)
        print("Agent A Thinking...")
        response_a_2 = self.chain_a.invoke({"opponent_last_message": response_b_1, "context": context})['text'].strip()
        transcript.append({"sender": "Agent A", "message": response_a_2})
        
        # Arbitrator / Summary (Simple Heuristic for demo)
        if "i will move" in response_a_2.lower() or "maneuvering" in response_a_2.lower():
            resolution = "Agent A agreed to maneuver."
        elif "i will move" in response_b_1.lower() or "maneuvering" in response_b_1.lower():
            resolution = "Agent B agreed to maneuver."
        else:
            resolution = "Negotiation Stalemate. Ground Control Intervention Required."
            
        transcript.append({"sender": "result", "message": resolution})
        
        return transcript

if __name__ == "__main__":
    dip = DiplomatSystem()
    result = dip.run_negotiation()
    for line in result:
        print(f"{line['sender']}: {line['message']}")
