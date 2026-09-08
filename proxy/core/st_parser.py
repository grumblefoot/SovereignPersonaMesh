import logging
from typing import List, Any

logger = logging.getLogger(__name__)

def parse_sillytavern_context(messages: List[Any]) -> str:
    """
    Parses an array of SillyTavern ChatCompletion messages and extracts 
    only the essential world-building context, dropping noise like generic 
    prompts and example chats.
    
    Args:
        messages: List of ChatCompletionMessage objects or dicts.
    Returns:
        A condensed string containing only character, scenario, and history.
    """
    kept_blocks = []
    drop_next = False
    
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            role = msg.get("role", "")
        else:
            content = getattr(msg, "content", "")
            role = getattr(msg, "role", "")
            
        if not content:
            continue
            
        # Drop user messages completely to avoid polluting core character extraction with user nonsense
        if role == "user":
            continue
            
        content_stripped = content.strip()
        
        # Drop if it was flagged by a previous Example Chat marker
        if drop_next:
            drop_next = False
            continue
            
        # Identify Example Chat marker
        if "[Example Chat]" in content_stripped:
            drop_next = True
            continue
            
        # Identify "Start a new Chat" marker
        if "[Start a new Chat]" in content_stripped:
            continue
            
        # Identify generic SillyTavern system prompts
        if role == "system" and "Write" in content_stripped and "next reply" in content_stripped:
            continue
            
        # Keep everything else
        kept_blocks.append(content_stripped)
        
    final_context = "\n\n".join(kept_blocks)
    return final_context
