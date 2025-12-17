import difflib

def compare(original: str, corrected: str) -> str:
    """
    Word-level difference between original and corrected text
    """
    diff = difflib.ndiff(original.split(), corrected.split())
    return "\n".join(diff)
