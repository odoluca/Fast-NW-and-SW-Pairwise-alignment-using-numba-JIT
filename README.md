# Fast-NW-and-SW-Pairwise-alignment-using-numba-JIT
This project includes Needleman-Wunsch and Smith-Waterman algorithms and their afine gap variations (Gotoh) written to work with Cython, PyPy and Numba. Numba JIT shows greater performance and requires Numpy module. Should you not have Numba, remove the "@jit()" decorations and numba import. PyPy and Cython works best with pairwise_standard.py.

For Best performance use pairwise_jit.py.

"local" is for Smith-Waterman algorithm. "global" is for Needleman-Wunsch.

method name extentions:
 
"ms": same match/mismatch scores for all residues. seperate gap open/extend penalties
"ds": uses a substitution matrix for different mismatch penalties between different residues. seperate gap open/extend penalties
  
"_align": report an alignment of sequences. If not used, it returns only the best score. (best score is useful for construction of distance matrix and phylogenetic tree construction)
  
 ===The use example for pairwise_JIT.py===
 
>>>a=convert_to_numeric("ATCTAGTCA")
>>>b=convert_to_numeric("ATCTAGTCACGTAG")
>>>res=globalms_align(a,b,1,-1,-1,-2)
>>>res
(array([ 1,  0,  3,  0,  1,  2,  0,  3,  1, -1, -1, -1, -1, -1], dtype=int64), array([1, 0, 3, 0, 1, 2, 0, 3, 1, 3, 2, 0, 1, 2], dtype=int64), 4.0)>>>report_alignment(res)
>>>report_alignment(res)
ATCTAGTCA-----
ATCTAGTCACGTAG
 score: 4.0
 
 ===The use example for pairwise_standard.py=== !SLOWER!
 
>>>a="ATCTAGTCA"
>>>b="ATCTAGTCACGTAG"
>>>res=globalms_align(a,b,1,-1,-1,-2)
>>>res
(['A', 'T', 'C', 'T', 'A', 'G', 'T', 'C', 'A', -1, -1, -1, -1, -1], ['A', 'T', 'C', 'T', 'A', 'G', 'T', 'C', 'A', 'C', 'G', 'T', 'A', 'G'], 4)
>>>report_alignment(res)
ATCTAGTCA-----
ATCTAGTCACGTAG
 score: 4.0
