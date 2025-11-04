# src/utils/fragment_parser.py
import re
from typing import Dict, List, Tuple
import pandas as pd


class FragmentAnnotationParser:
    """Parse UniProt fragment annotation fields."""
    
    # Fragment type labels (multilabel)
    N_TERMINAL = 'n_terminal'
    C_TERMINAL = 'c_terminal'
    INTERNAL = 'internal'
    
    def __init__(self, n_terminal_threshold: int = 10, c_terminal_threshold: int = 10):
        """
        Args:
            n_terminal_threshold: Position threshold for N-terminal fragments
            c_terminal_threshold: Distance from end for C-terminal fragments
        """
        self.n_term_thresh = n_terminal_threshold
        self.c_term_thresh = c_terminal_threshold
    
    def parse_non_ter(self, non_ter_str: str) -> List[int]:
        """
        Extract positions from NON_TER annotations.
        
        Args:
            non_ter_str: String like "NON_TER 1; /evidence=... NON_TER 233"
            
        Returns:
            List of position integers
        """
        if pd.isna(non_ter_str) or not non_ter_str.strip():
            return []
        
        # Extract all numbers after NON_TER
        positions = re.findall(r'NON_TER\s+(\d+)', str(non_ter_str))
        return [int(p) for p in positions]
    
    def parse_non_cons(self, non_cons_str: str) -> List[Tuple[int, int]]:
        """
        Extract gap ranges from NON_CONS annotations.
        
        Args:
            non_cons_str: String like "NON_CONS 52..53"
            
        Returns:
            List of (start, end) tuples representing gaps
        """
        if pd.isna(non_cons_str) or not non_cons_str.strip():
            return []
        
        # Extract ranges like "52..53"
        ranges = re.findall(r'NON_CONS\s+(\d+)\.\.(\d+)', str(non_cons_str))
        return [(int(start), int(end)) for start, end in ranges]
    
    def classify_fragment(
        self, 
        non_ter_positions: List[int],
        non_cons_gaps: List[Tuple[int, int]],
        sequence_length: int
    ) -> Dict[str, bool]:
        """
        Classify fragment type based on annotations and sequence length.
        """
        labels = {
            self.N_TERMINAL: False,
            self.C_TERMINAL: False,
            self.INTERNAL: False
        }
        
        # Internal gaps from NON_CONS
        if non_cons_gaps:
            labels[self.INTERNAL] = True
        
        # Classify NON_TER positions
        if not non_ter_positions:
            # If no NON_TER/NON_CONS, it's a fragment but we don't know type.
            # But the project implies these fields define the type.
            # We will assume if it's a fragment with no info, it's one or the other.
            # For this parser, we only label what's explicit.
            return labels

        for pos in non_ter_positions:
            # N-terminal if position is near start
            if pos <= self.n_term_thresh:
                labels[self.N_TERMINAL] = True
            
            # C-terminal if position is near end
            # Use >= to catch 1-based indexing (e.g., length 250, NON_TER 250)
            if sequence_length > 0 and (sequence_length - pos) < self.c_term_thresh:
                labels[self.C_TERMINAL] = True
            
            # Internal if position is in the middle
            if pos > self.n_term_thresh and (sequence_length - pos) >= self.c_term_thresh:
                labels[self.INTERNAL] = True
        
        return labels