# Fast-NW-and-SW-Pairwise-alignment-using-numba-JIT
This project includes Needleman-Wunsch and Smith-Waterman algorithms and their afine gap variations (Gotoh) written to work with Cython, PyPy and Numba. Numba JIT shows greater performance.

For Best performance use gotoh_jit.py to get only the best score and use gotoh_jit_traceback to get the best alignment

"local" is for Smith-Waterman algorithm. "global" is for Needleman-Wunsch.

method name extentions:
 
"ms": same match/mismatch scores for all residues. seperate gap open/extend penalties
"ds": uses a substitution matrix for different mismatch penalties between different residues. seperate gap open/extend penalties
"dd": NOT COMPLETE YET
  
"_align": report an alignment of sequences. If not used, it returns only the best score. (best score is useful for construction of distance matrix and phylogenetic tree construction)
  
  
