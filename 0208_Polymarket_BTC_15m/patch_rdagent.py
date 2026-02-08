import os
import sys
from pathlib import Path

def patch_litellm():
    """
    Patch rdagent/oai/backend/litellm.py to remove json_mode and response_format
    parameters when using Gemini models, as they cause 400 Bad Request errors.
    """
    target_file = Path(".venv_rdagent/lib/python3.10/site-packages/rdagent/oai/backend/litellm.py")
    if not target_file.exists():
        print(f"❌ Target file not found: {target_file}")
        return False
    
    print(f"Checking patch for: {target_file}")
    content = target_file.read_text(encoding="utf-8")
    
    # Patch logic
    patch_code = """
        # PATCH: Gemini via LiteLLM often fails with json_mode=True or response_format
        if "gemini" in model.lower():
            if "response_format" in kwargs:
                del kwargs["response_format"]
            if "json_mode" in kwargs:
                del kwargs["json_mode"]
"""
    
    if "PATCH: Gemini via LiteLLM often fails" in content:
        print("✅ litellm.py is already patched.")
        return True

    # Look for the insertion point
    target_str = '        model = complete_kwargs["model"]'
    if target_str not in content:
        print("❌ Could not find insertion point in litellm.py")
        return False
        
    new_content = content.replace(target_str, target_str + patch_code)
    target_file.write_text(new_content, encoding="utf-8")
    print("✅ Successfully patched litellm.py")
    return True

def patch_llm_conf():
    """
    Patch rdagent/oai/llm_conf.py to increase default retry_wait_seconds to 60s
    to handle Gemini's strict rate limits (requires ~38s cooldown).
    """
    target_file = Path(".venv_rdagent/lib/python3.10/site-packages/rdagent/oai/llm_conf.py")
    if not target_file.exists():
        print(f"❌ Target file not found: {target_file}")
        return False

    print(f"Checking patch for: {target_file}")
    content = target_file.read_text(encoding="utf-8")
    
    # Check if already 60
    if "retry_wait_seconds: int = 60" in content:
        print("✅ llm_conf.py is already patched (retry_wait_seconds=60).")
        return True
        
    if "retry_wait_seconds: int = 1" in content:
        new_content = content.replace("retry_wait_seconds: int = 1", "retry_wait_seconds: int = 60")
        target_file.write_text(new_content, encoding="utf-8")
        print("✅ Successfully patched llm_conf.py (retry_wait_seconds=60)")
        return True
    
    print("⚠️ Could not find 'retry_wait_seconds: int = 1' to patch. It might have been modified manually.")
    return False

if __name__ == "__main__":
    print("--- Applying RD-Agent Patches for Gemini ---")
    p1 = patch_litellm()
    p2 = patch_llm_conf()
    
    if p1 and p2:
        print("--- All Patches Applied Successfully ---")
        sys.exit(0)
    else:
        print("--- Some Patches Failed ---")
        sys.exit(1)
