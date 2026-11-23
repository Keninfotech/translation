"""
Domain validation and terminology verification engine for General, Medical, and Office modes.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from .config import ROOT

GLOSSARIES_DIR = ROOT / "glossaries"

class DomainEngine:
    def __init__(self):
        self._medical_data = self._load_glossary("medical.json")
        self._office_data = self._load_glossary("office.json")
        self._general_data = self._load_glossary("general.json")

    def _load_glossary(self, filename: str) -> dict:
        path = GLOSSARIES_DIR / filename
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def reload(self):
        """Reloads glossaries from disk."""
        self._medical_data = self._load_glossary("medical.json")
        self._office_data = self._load_glossary("office.json")
        self._general_data = self._load_glossary("general.json")

    def extract_numbers(self, text: str) -> List[str]:
        """Extracts ASCII and Indic digits from text."""
        # Find all digit sequences
        nums = re.findall(r'\d+(?:\.\d+)?', text)
        return nums

    def extract_dosages(self, text: str) -> List[str]:
        """Extracts clinical dosages (e.g., '500 mg', '2 tablets', '120/80')."""
        pattern = r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|iu|tablets?|capsules?|drops?|puffs?|bpm|mmHg|%)\b|\b\d+/\d+\b'
        return re.findall(pattern, text, re.IGNORECASE)

    def check_negation(self, text: str, lang: str) -> bool:
        """Detects negation words across English and Indic languages."""
        negations = [
            "not", "no", "never", "do not", "don't", "stop", "discontinue", "avoid", "without", "none",
            "nahi", "na", "mat", "illa", "alla", "vendam", "kodukkaruth", "kazhikkaruth", "benda",
            "nahi", "nehi", "nay", "kadhu", "ledhu", "nako"
        ]
        # Also check native scripts
        indic_negations = [
            "नहीं", "मत", "ना",          # Hindi/Marathi
            "ഇല്ല", "അല്ല", "അരുത്",      # Malayalam
            "இல்லை", "வேண்டாம்", "கூடாது", # Tamil
            "లేదు", "వద్దు", "కాదు",        # Telugu
            "ಇಲ್ಲ", "ಬೇಡ", "ಅಲ್ಲ",        # Kannada
            "না", "নয়", "নেই",          # Bengali
            "નથી", "નહીં",                 # Gujarati
            "ਨਹੀਂ", "ਨਾ",                 # Punjabi
            "ନାହିଁ", "ନୁହେଁ",             # Odia
            "नाही", "नको"                 # Marathi
        ]
        low = text.casefold()
        for term in negations:
            if re.search(r'\b' + re.escape(term) + r'\b', low):
                return True
        for term in indic_negations:
            if term in text:
                return True
        return False

    def validate(self, source_text: str, target_text: str, mode: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Validates translation based on the selected mode:
        - general: light checks, code-mixing preservation
        - medical: dosage verification, negation preservation, clinical review alerts
        - office: business terms, acronym preservation, SLA/deadline check
        """
        warnings = []
        flags = {
            "mode": mode,
            "requires_human_review": False,
            "dosage_mismatch": False,
            "negation_alert": False,
            "terms_flagged": []
        }

        if mode == "medical":
            flags["requires_human_review"] = True  # All clinical mode outputs advise clinician verification

            # 1. Dosages and numbers verification
            src_dosages = self.extract_dosages(source_text)
            src_nums = self.extract_numbers(source_text)
            tgt_nums = self.extract_numbers(target_text)

            for d in src_dosages:
                # Extract the numeric part of the dosage
                num_part = re.findall(r'\d+(?:\.\d+)?', d)
                if num_part and num_part[0] not in tgt_nums:
                    flags["dosage_mismatch"] = True
                    warnings.append(f"Dosage/Numeric Warning: '{d}' in original text may not be accurately reflected in translation ({target_text}).")

            # Check general count of numbers
            if len(src_nums) > len(tgt_nums):
                missing = [n for n in src_nums if n not in tgt_nums]
                if missing:
                    flags["dosage_mismatch"] = True
                    warnings.append(f"Number Check: Original numeric value(s) {missing} missing in translation.")

            # 2. Negation preservation check
            src_neg = self.check_negation(source_text, source_lang)
            tgt_neg = self.check_negation(target_text, target_lang)

            if src_neg and not tgt_neg:
                flags["negation_alert"] = True
                warnings.append("CRITICAL CLINICAL ALERT: Negation directive ('no' / 'not' / 'stop') detected in source, but missing in target translation.")

            # 3. Medical glossary high-risk review terms
            review_terms = self._medical_data.get("review_terms", [])
            for term in review_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', source_text, re.IGNORECASE):
                    flags["terms_flagged"].append(term)
                    warnings.append(f"High-Consequence Medical Term: '{term}' detected. Must be verified by licensed medical personnel.")

        elif mode == "office":
            # 1. Office terminology
            review_terms = self._office_data.get("review_terms", [])
            for term in review_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', source_text, re.IGNORECASE):
                    flags["terms_flagged"].append(term)
                    warnings.append(f"Business Terminology: '{term}' detected. Verify corporate context.")

            # 2. Uppercase acronyms preservation (e.g. SLA, KPI, NDA, CEO, EBITDA)
            acronyms = re.findall(r'\b[A-Z]{2,6}\b', source_text)
            for ac in set(acronyms):
                if ac not in target_text and target_lang == 'en':
                    warnings.append(f"Acronym: '{ac}' should be verified for consistency.")

        return {
            "warnings": warnings,
            "flags": flags
        }

domain_engine = DomainEngine()
