# app/services/sequential_executor.py

import time
from typing import Optional, Type
from pydantic import BaseModel
from app.services.agent_wrapper import AgentWrapper


class SequentialExecutor:
    """
    Executes a single agent call (no parallel execution).
    Designed for free tier API limits.
    """
    def __init__(self, agent_wrapper: AgentWrapper):
        if not agent_wrapper:
            raise ValueError("AgentWrapper must be injected into SequentialExecutor.")
        self.agent_wrapper = agent_wrapper
    
    async def run_single_execution(
        self, 
        prompt: str,
        response_schema: Optional[Type[BaseModel]] = None
    ) -> BaseModel:
        """
        Execute a single agent call without parallel redundancy.
        
        Args:
            prompt: The prompt to execute
            response_schema: Pydantic model for response validation
        
        Returns:
            Validated Pydantic model instance
        """
        try:
            # Pass the schema to the agent wrapper
            raw_text = self.agent_wrapper.execute(prompt, response_schema=response_schema)
            
            # Schema validate
            if response_schema:
                validated_model = response_schema.model_validate_json(raw_text)
                return validated_model
            else:
                # Return raw text if no schema
                return raw_text
                
        except Exception as e:
            print(f"❌ Execution Failed: {type(e).__name__} - {str(e)}")
            raise