import json

with open("proxy/rag/res/strings.json", "r") as f:
    data = json.load(f)

if "rag" not in data:
    data["rag"] = {}

data["rag"]["lore_extractor"] = {
    "initial_rules_prompt": """Analyze the following context text (which contains character cards, lore, and user personas). Your goal is to establish the Core Identity of the Non-Player Character (NPC) and the World Scenario by extracting a list of GM rules (invariants/triggers) that should govern the game/master narrative.

CRITICAL INSTRUCTION:
1. Focus ONLY on the Non-Player Characters (NPCs), the world, the environment, and the overarching scenario.
2. Do NOT extract rules about the User or Player character's physical description, clothing, or persona.
3. You MUST extract rules regarding the NPC's core identity. Look closely for:
   - Character Goals & Motivations (e.g., "Arvenia is driven by a desire to reclaim her family's honor").
   - Deep Personality Traits & Quirks (e.g., "Arvenia is fiercely independent and refuses charity").
   - NPC Physical Traits & Appearance (e.g., "Arvenia has silver hair, is slender, and wears a blue dress").
   - Overarching Scenario Parameters (e.g., "The tavern is located in a dangerous slum where theft is common").
   Formulate these as 'invariant' rules (always true) or 'trigger' rules (if X happens, NPC does Y).

Output ONLY a valid JSON list of objects. Each object must strictly contain:
- "rule_text": string describing the rule
- "rule_type": string (must be either "invariant" or "trigger")

Context:
{context_text}
""",
    "periodic_rules_prompt": """Analyze the following recent chat messages and propose any new GM rules, updates to existing rules, or narrative invariants/triggers that should be tracked.
Output ONLY a valid JSON list of objects. Each object must strictly contain:
- "rule_text": string describing the rule
- "rule_type": string (must be either "invariant" or "trigger")

Recent Messages:
{recent_messages}
"""
}

with open("proxy/rag/res/strings.json", "w") as f:
    json.dump(data, f, indent=2)

print("Updated strings.json")
