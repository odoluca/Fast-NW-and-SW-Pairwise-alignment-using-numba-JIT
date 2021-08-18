import numpy as np
from numba import jit,cuda

# noinspection PyPep8Naming
@jit(nopython=True,cache=True)
def globalms(A, B, match , mismatch , gap_open , gap_extend ,penalize_extend_when_opening=False):
    #""" Initializes and fills up the matrices and calculates the alignment score.  """
    n = len(A)
    m = len(B)

    neg_inf = -np.inf

    # s function
    def s(x, y):
        if x == y:
            return match
        else:
            return mismatch

    # g function

    def g(k):
        if penalize_extend_when_opening:
            return gap_open + gap_extend*k
        else:
            return gap_open+gap_extend*(k-1)

    # construct and initialize the matrices
    D = np.zeros((n+1, m+1))
    D[0, 0] = 0
    D[0, 1] = g(1)
    D[1, 0] = g(1)
    for i in range(2, n+1):
        D[i, 0] = D[i-1, 0] +gap_extend
    for j in range(2, m+1):
        D[0, j] = D[0, j-1] +gap_extend

    P = np.empty((n + 1, m + 1))
    for i in range(1, n+1):
        P[i, 0] = neg_inf
    for j in range(1, m + 1):
        P[0, j] = neg_inf

    Q = np.empty((n+1, m+1))
    for j in range(1, m+1):
        Q[0, j] = neg_inf
    for i in range(1, n+1):
        Q[i, 0] = neg_inf

    # fill up the rest of the matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            P[i, j] = max( D[i-1, j] + g(1), P[i-1, j] + gap_extend )
            Q[i, j] = max( D[i, j-1] + g(1), Q[i, j-1] + gap_extend )
            D[i, j] = max(D[i - 1, j - 1] + s(A[i - 1], B[j - 1]), P[i, j], Q[i, j])

    Score,level=D[n,m],0
    if P[n,m]>Score:
        Score, level = P[n, m], 0
    if Q[n,m]>Score:
        Score, level = Q[n, m], 0
    return Score
    

@jit(nopython=True,cache=True)
def globalds(A, B, subs_mat, gap_open = -3, gap_extend = -1,penalize_extend_when_opening=False):
    """ Initializes and fills up the matrices and calculates the alignment score.  """

    n = len(A)
    m = len(B)

    neg_inf = -np.inf

    # s function
    def s(x, y):
        return subs_mat[x, y]


    # g function
    def g(k):
        if penalize_extend_when_opening:
            return gap_open + gap_extend*k
        else:
            return gap_open + gap_extend*(k-1)

    # construct and initialize the matrices
    D = np.zeros((n+1, m+1))
    D[0, 1] = g(1)
    D[1, 0] = g(1)
    for i in range(2, n+1):
        D[i, 0] = D[i-1, 0] +gap_extend
    for j in range(2, m+1):
        D[0, j] = D[0, j-1] +gap_extend

    P = np.empty((n+1, m+1))
    for i in range(1, n+1):
        P[i, 0] = neg_inf
    for j in range(1, m + 1):
        P[0, j] = neg_inf

    Q = np.empty((n+1, m+1))
    for j in range(1, m+1):
        Q[0, j] = neg_inf
    for i in range(1, n+1):
        Q[i, 0] = neg_inf

    # fill up the rest of the matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            P[i, j] = max( D[i-1, j] + g(1), P[i-1, j] + gap_extend )
            Q[i, j] = max( D[i, j-1] + g(1), Q[i, j-1] + gap_extend )
            D[i, j] = max( D[i - 1, j - 1] + s(A[i -1], B[j -1]), P[i, j], Q[i, j])

    Score,level=D[n,m],0
    if P[n,m]>Score:
        Score, level = P[n, m], 0
    if Q[n,m]>Score:
        Score, level = Q[n, m], 0
    return Score
    
@jit(nopython=True,cache=True)
def localms(A, B, match = 1, mismatch = -1, gap_open = -3, gap_extend = -1,penalize_extend_when_opening=False):
    """ Initializes and fills up the matrices and calculates the alignment score.  """

    n = len(A)
    m = len(B)

    neg_inf = -np.inf

    # s function
    def s(x, y):
        if x == y:
            return match
        else:
            return mismatch

    # g function
    def g(k):
        if penalize_extend_when_opening:
            return gap_open + gap_extend*k
        else:
            return gap_open+gap_extend*(k-1)

    # construct and initialize the matrices
    D = np.zeros((n+1, m+1))

    P = np.empty((n+1, m+1))
    for i in range(1, n+1):
        P[i, 0] = neg_inf
    for j in range(1, m + 1):
        P[0, j] = neg_inf

    Q = np.empty((n+1, m+1))
    for j in range(1, m+1):
        Q[0, j] = neg_inf
    for i in range(1, n+1):
        Q[i, 0] = neg_inf

    # fill up the rest of the matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            P[i, j] = max( D[i-1, j] + g(1), P[i-1, j] + gap_extend )
            Q[i, j] = max( D[i, j-1] + g(1), Q[i, j-1] + gap_extend )
            D[i, j] = max(0,D[i - 1, j - 1] + s(A[i - 1], B[j - 1]), P[i, j], Q[i, j])

    # Score calculation
    max_n, max_m = 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if D[i, j] > D[max_n, max_m]:  # and (i == n or j == m):
                max_n, max_m = i, j
    Score=D[max_n,max_m]
    return Score
    


@jit(nopython=True,cache=True)
def localds(A, B, subs_mat, gap_open = -3, gap_extend = -1,penalize_extend_when_opening=False):
    """ Initializes and fills up the matrices and calculates the alignment score.  """

    n = len(A)
    m = len(B)

    neg_inf = -np.inf

    # s function
    def s(x, y):
        return subs_mat[x, y]


    # g function
    def g(k):
        if penalize_extend_when_opening:
            return gap_open + gap_extend*k
        else:
            return gap_open + gap_extend*(k-1)

    # construct and initialize the matrices
    D = np.zeros((n+1, m+1))

    P = np.empty((n+1, m+1))
    for i in range(1, n+1):
        P[i, 0] = neg_inf
    for j in range(1, m + 1):
        P[0, j] = neg_inf

    Q = np.empty((n+1, m+1))
    for j in range(1, m+1):
        Q[0, j] = neg_inf
    for i in range(1, n+1):
        Q[i, 0] = neg_inf

    # fill up the rest of the matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            P[i, j] = max( D[i-1, j] + g(1), P[i-1, j] + gap_extend )
            Q[i, j] = max( D[i, j-1] + g(1), Q[i, j-1] + gap_extend )
            D[i, j] = max(0,D[i - 1, j - 1] + s(A[i - 1], B[j - 1]), P[i, j], Q[i, j])

    # Score calculation
    max_n, max_m = 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if D[i, j] > D[max_n, max_m]:  # and (i == n or j == m):
                max_n, max_m = i, j
    Score=D[max_n,max_m]
    return Score
    
@jit(nopython=True,cache=True)
def globalms_align(A, B, match , mismatch , gap_open , gap_extend ,penalize_extend_when_opening=False):
    #""" Initializes and fills up the matrices and calculates the alignment score.  """
    n = len(A)
    m = len(B)

    neg_inf = -np.inf

    # s function
    def s(x, y):
        if x == y:
            return match
        else:
            return mismatch

    # g function

    def g(k):
        if penalize_extend_when_opening:
            return gap_open + gap_extend*k
        else:
            return gap_open+gap_extend*(k-1)

    # construct and initialize the matrices
    D = np.zeros((n+1, m+1))
    D[0, 0] = 0
    D[0, 1] = g(1)
    D[1, 0] = g(1)
    for i in range(2, n+1):
        D[i, 0] = D[i-1, 0] +gap_extend
    for j in range(2, m+1):
        D[0, j] = D[0, j-1] +gap_extend

    P = np.empty((n + 1, m + 1))
    for i in range(1, n+1):
        P[i, 0] = neg_inf
    for j in range(1, m + 1):
        P[0, j] = neg_inf

    Q = np.empty((n+1, m+1))
    for j in range(1, m+1):
        Q[0, j] = neg_inf
    for i in range(1, n+1):
        Q[i, 0] = neg_inf

    #keep all directions here
    dir_on_D=np.zeros((n+1,m+1))
    dir_on_P = np.zeros((n + 1, m + 1))
    dir_on_Q = np.zeros((n + 1, m + 1))
    for j in range(1, m+1):
        dir_on_D[0,j]=100
        dir_on_Q[0, j] = 1
    for i in range(1, n + 1):
        dir_on_D[i,0]=10
        dir_on_P[i, 0] = 1

    # fill up the rest of the matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            P[i, j] = max( D[i-1, j] + g(1), P[i-1, j] + gap_extend )
            Q[i, j] = max( D[i, j-1] + g(1), Q[i, j-1] + gap_extend )
            D[i, j] = max(D[i - 1, j - 1] + s(A[i - 1], B[j - 1]), P[i, j], Q[i, j])

            #for traceback:
            dir_on_P[i,j]=D[i-1, j] + g(1)< P[i-1, j] + gap_extend
            dir_on_Q[i,j]=D[i, j-1] + g(1)< Q[i, j-1] + gap_extend
            if D[i, j]==D[i - 1, j - 1] + s(A[i - 1], B[j - 1]):
                dir_on_D[i,j]=1
            if D[i, j] ==P[i, j]:
                dir_on_D[i,j]=dir_on_D[i,j]+10
            if D[i, j] == Q[i, j]:
                dir_on_D[i, j]=dir_on_D[i,j]+100

    Score,level=D[n,m],0
    if P[n,m]>Score:
        Score, level = P[n, m], 0
    if Q[n,m]>Score:
        Score, level = Q[n, m], 0

    al1=[]
    al2=[]
    i,j=n,m
    #Extract one of the best alignments
    while i!=0 or j!=0:
        if level==0:
            if (dir_on_D[i,j]//10)%10==1:  level=1
            elif (dir_on_D[i,j]//100)%10==1: level=2
            elif dir_on_D[i, j] % 10 == 1:
                i,j=i-1,j-1
                al1.append(A[i])
                al2.append(B[j])
        elif level==1:
            if dir_on_P[i,j]==0: level=0
            i=i-1
            al1.append(A[i])
            al2.append(-1)
        elif level==2:
            if dir_on_Q[i,j]==0: level=0
            j=j-1
            al1.append(-1)
            al2.append(B[j])

    return np.array(al1[::-1]),np.array(al2[::-1]),Score #orientation corrected arrays and best score

@jit(nopython=True,cache=True)
def globalds_align(A, B, subs_mat, gap_open = -3, gap_extend = -1,penalize_extend_when_opening=False):
    """ Initializes and fills up the matrices and calculates the alignment score.  """

    n = len(A)
    m = len(B)

    neg_inf = -np.inf

    # s function
    def s(x, y):
        return subs_mat[x, y]


    # g function
    def g(k):
        if penalize_extend_when_opening:
            return gap_open + gap_extend*k
        else:
            return gap_open + gap_extend*(k-1)

    # construct and initialize the matrices
    D = np.zeros((n+1, m+1))
    D[0, 1] = g(1)
    D[1, 0] = g(1)
    for i in range(2, n+1):
        D[i, 0] = D[i-1, 0] +gap_extend
    for j in range(2, m+1):
        D[0, j] = D[0, j-1] +gap_extend

    P = np.empty((n+1, m+1))
    for i in range(1, n+1):
        P[i, 0] = neg_inf
    for j in range(1, m + 1):
        P[0, j] = neg_inf

    Q = np.empty((n+1, m+1))
    for j in range(1, m+1):
        Q[0, j] = neg_inf
    for i in range(1, n+1):
        Q[i, 0] = neg_inf

    #keep all directions here
    dir_on_D=np.zeros((n+1,m+1))
    dir_on_P = np.zeros((n + 1, m + 1))
    dir_on_Q = np.zeros((n + 1, m + 1))
    for j in range(1, m+1):
        dir_on_D[0,j]=100
        dir_on_Q[0, j] = 1
    for i in range(1, n + 1):
        dir_on_D[i,0]=10
        dir_on_P[i, 0] = 1

    # fill up the rest of the matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            P[i, j] = max( D[i-1, j] + g(1), P[i-1, j] + gap_extend )
            Q[i, j] = max( D[i, j-1] + g(1), Q[i, j-1] + gap_extend )
            D[i, j] = max(D[i - 1, j - 1] + s(A[i -1], B[j -1]), P[i, j], Q[i, j])

            #for traceback:
            dir_on_P[i,j]=   D[i-1, j] + g(1)< P[i-1, j] + gap_extend
            dir_on_Q[i,j]=D[i, j-1] + g(1)< Q[i, j-1] + gap_extend
            if D[i, j]==D[i - 1, j - 1] + s(A[i - 1], B[j - 1]):
                dir_on_D[i,j]=1
            if D[i, j] ==P[i, j]:
                dir_on_D[i,j]=dir_on_D[i,j]+10
            if D[i, j] == Q[i, j]:
                dir_on_D[i, j]=dir_on_D[i,j]+100

    Score,level=D[n,m],0
    if P[n,m]>Score:
        Score, level = P[n, m], 0
    if Q[n,m]>Score:
        Score, level = Q[n, m], 0

    al1=[]
    al2=[]
    i,j=n,m
    #Extract one of the best alignments
    while i!=0 or j!=0:
        if level==0:
            if (dir_on_D[i,j]//10)%10==1:  level=1
            elif (dir_on_D[i,j]//100)%10==1: level=2
            elif dir_on_D[i, j] % 10 == 1:
                i,j=i-1,j-1
                al1.append(A[i])
                al2.append(B[j])
        elif level==1:
            if dir_on_P[i,j]==0: level=0
            i=i-1
            al1.append(A[i])
            al2.append(-1)
        elif level==2:
            if dir_on_Q[i,j]==0: level=0
            j=j-1
            al1.append(-1)
            al2.append(B[j])

    return np.array(al1[::-1]),np.array(al2[::-1]),Score #orientation corrected arrays and best score


@jit(nopython=True,cache=True)
def localms_align(A, B, match = 1, mismatch = -1, gap_open = -3, gap_extend = -1,penalize_extend_when_opening=False):
    """ Initializes and fills up the matrices and calculates the alignment score.  """

    n = len(A)
    m = len(B)

    neg_inf = -np.inf

    # s function
    def s(x, y):
        if x == y:
            return match
        else:
            return mismatch

    # g function
    def g(k):
        if penalize_extend_when_opening:
            return gap_open + gap_extend*k
        else:
            return gap_open+gap_extend*(k-1)

    # construct and initialize the matrices
    D = np.zeros((n+1, m+1))

    P = np.empty((n+1, m+1))
    for i in range(1, n+1):
        P[i, 0] = neg_inf
    for j in range(1, m + 1):
        P[0, j] = neg_inf

    Q = np.empty((n+1, m+1))
    for j in range(1, m+1):
        Q[0, j] = neg_inf
    for i in range(1, n+1):
        Q[i, 0] = neg_inf

    #keep all directions here
    dir_on_D=np.zeros((n+1,m+1))
    dir_on_P = np.zeros((n + 1, m + 1))
    dir_on_Q = np.zeros((n + 1, m + 1))
    for j in range(1, m+1):
        dir_on_D[0,j]=100
        dir_on_Q[0, j] = 1
    for i in range(1, n + 1):
        dir_on_D[i,0]=10
        dir_on_P[i, 0] = 1

    # fill up the rest of the matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            P[i, j] = max( D[i-1, j] + g(1), P[i-1, j] + gap_extend )
            Q[i, j] = max( D[i, j-1] + g(1), Q[i, j-1] + gap_extend )
            D[i, j] = max(0,D[i - 1, j - 1] + s(A[i - 1], B[j - 1]), P[i, j], Q[i, j])

            #for traceback:
            dir_on_P[i,j]=D[i-1, j] + g(1)< P[i-1, j] + gap_extend
            dir_on_Q[i,j]=D[i, j-1] + g(1)< Q[i, j-1] + gap_extend
            if D[i, j] == D[i - 1, j - 1] + s(A[i - 1], B[j - 1]):
                dir_on_D[i, j] = 1
            if D[i, j] == P[i, j]:
                dir_on_D[i, j] = dir_on_D[i, j] + 10
            if D[i, j] == Q[i, j]:
                dir_on_D[i, j] = dir_on_D[i, j] + 100

    # Score calculation
    max_n, max_m = 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if D[i, j] > D[max_n, max_m]:  # and (i == n or j == m):
                max_n, max_m = i, j
    Score=D[max_n,max_m]

    #Extract one of the best alignments
    level=0 #level starts at 0 when local
    al1 = []
    al2 = []
    i, j = max_n, max_m
    while D[i,j]>0:
        if level==0:
            if (dir_on_D[i,j]//10)%10==1:  level=1
            elif (dir_on_D[i,j]//100)%10==1: level=2
            elif dir_on_D[i, j] % 10 == 1:
                i,j=i-1,j-1
                al1.append(A[i])
                al2.append(B[j])
        elif level==1:
            if dir_on_P[i,j]==0: level=0
            i=i-1
            al1.append(A[i])
            al2.append(-1)
        elif level==2:
            if dir_on_Q[i,j]==0: level=0
            j=j-1
            al1.append(-1)
            al2.append(B[j])
    return np.array(al1[::-1]), np.array(al2[::-1]), Score #orientation corrected arrays and best score


@jit(nopython=True,cache=True)
def localds_align(A, B, subs_mat, gap_open = -3, gap_extend = -1,penalize_extend_when_opening=False):
    """ Initializes and fills up the matrices and calculates the alignment score.  """

    n = len(A)
    m = len(B)

    neg_inf = -np.inf

    # s function
    def s(x, y):
        return subs_mat[x, y]


    # g function
    def g(k):
        if penalize_extend_when_opening:
            return gap_open + gap_extend*k
        else:
            return gap_open + gap_extend*(k-1)

    # construct and initialize the matrices
    D = np.zeros((n+1, m+1))

    P = np.empty((n+1, m+1))
    for i in range(1, n+1):
        P[i, 0] = neg_inf
    for j in range(1, m + 1):
        P[0, j] = neg_inf

    Q = np.empty((n+1, m+1))
    for j in range(1, m+1):
        Q[0, j] = neg_inf
    for i in range(1, n+1):
        Q[i, 0] = neg_inf

    #keep all directions here
    dir_on_D=np.zeros((n+1,m+1))
    dir_on_P = np.zeros((n + 1, m + 1))
    dir_on_Q = np.zeros((n + 1, m + 1))
    for j in range(1, m+1):
        dir_on_D[0,j]=100
        dir_on_Q[0, j] = 1
    for i in range(1, n + 1):
        dir_on_D[i,0]=10
        dir_on_P[i, 0] = 1

    # fill up the rest of the matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            P[i, j] = max( D[i-1, j] + g(1), P[i-1, j] + gap_extend )
            Q[i, j] = max( D[i, j-1] + g(1), Q[i, j-1] + gap_extend )
            D[i, j] = max(0,D[i - 1, j - 1] + s(A[i - 1], B[j - 1]), P[i, j], Q[i, j])

            #for traceback:
            dir_on_P[i,j]=   D[i-1, j] + g(1)< P[i-1, j] + gap_extend
            dir_on_Q[i,j]=D[i, j-1] + g(1)< Q[i, j-1] + gap_extend
            if D[i, j] == D[i - 1, j - 1] + s(A[i - 1], B[j - 1]):
                dir_on_D[i, j] = 1
            if D[i, j] == P[i, j]:
                dir_on_D[i, j] = dir_on_D[i, j] + 10
            if D[i, j] == Q[i, j]:
                dir_on_D[i, j] = dir_on_D[i, j] + 100

    # Score calculation
    max_n, max_m = 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if D[i, j] > D[max_n, max_m]:  # and (i == n or j == m):
                max_n, max_m = i, j
    Score=D[max_n,max_m]

    #Extract one of the best alignments
    level=0 #level starts at 0 when local
    al1 = []
    al2 = []
    i, j = max_n, max_m
    while D[i,j]>0:
        if level==0:
            if (dir_on_D[i,j]//10)%10==1:  level=1
            elif (dir_on_D[i,j]//100)%10==1: level=2
            elif dir_on_D[i, j] % 10 == 1:
                i,j=i-1,j-1
                al1.append(A[i])
                al2.append(B[j])
        elif level==1:
            if dir_on_P[i,j]==0: level=0
            i=i-1
            al1.append(A[i])
            al2.append(-1)
        elif level==2:
            if dir_on_Q[i,j]==0: level=0
            j=j-1
            al1.append(-1)
            al2.append(B[j])
    return np.array(al1[::-1]), np.array(al2[::-1]), Score #orientation corrected arrays and best score


def report_alignment(input,mol_type="auto",alphabet="auto"):
    # print(type(input))
    if type(input)==tuple:
        a,b,s=input
    else:
        print("Alignment score:",input)
        return
    # print(np.unique(a))
    if mol_type=="auto":
        if len(np.unique(a))>6 and len(np.unique(b))>6:
            mol_type="Protein"
        else:
            mol_type="DNA"

    if mol_type=="DNA":
        _alphabet = "TAGCN"
        if alphabet != "auto": _alphabet = alphabet
        conversion_dict = dict(zip(range(-1,5),"-"+_alphabet))

        # dictionary={0:"T",1:"A",2:"G",3:"C",4:"N",-1:"-"}
    if mol_type=="RNA":
        _alphabet = "UAGCN"
        if alphabet != "auto": _alphabet = alphabet
        conversion_dict = dict(zip(range(-1,5),"-"+_alphabet))

        # dictionary = {0: "U", 1: "A", 2: "G", 3: "C", 4: "N", -1: "-"}
    if mol_type=="Protein":
        _alphabet = "ARNDCQEGHILKMFPSTWYVBZX"
        if alphabet != "auto": _alphabet = alphabet
        conversion_dict = dict(zip(range(-1,len(_alphabet)),"-"+_alphabet))

    print("".join([conversion_dict[x] for x in a]))
    print("".join([conversion_dict[x] for x in b]))
    print(" score:",s)

    
def convert_to_numeric(seq,alphabet="auto",mol_type="auto"):
    seq=seq.upper().replace(" ", "").replace("\n", "").strip("*")
    set = [None] * len(seq)

    if mol_type=="auto":
        if all([(x in "ARNDCQEGHILKMFPSTWYVBZX") for x in seq]):
            mol_type="Protein"
        if all([x in "UAGCN"  for x in seq]):
            mol_type="RNA"
        if all([x in "TAGCN"  for x in seq]):
            mol_type="DNA"

    if mol_type=="DNA":
        _alphabet="TAGCN"
        if alphabet!="auto": _alphabet=alphabet
        conversion_dict=dict(zip(_alphabet,range(5)))
        for k in range(len(seq)):
            set[k]=conversion_dict[seq[k]]

    elif mol_type=="RNA":
        _alphabet="UAGCN"
        if alphabet!="auto": _alphabet=alphabet
        conversion_dict = dict(zip(_alphabet, range(5)))
        for k in range(len(seq)):
            set[k] = conversion_dict[seq[k]]

    # T:0,A:1,G:2,C:3,N:4,-1 is reserved for gap
    elif mol_type=="Protein":
        _alphabet="ARNDCQEGHILKMFPSTWYVBZX"
        if alphabet!="auto": _alphabet=alphabet
        conversion_dict = dict(zip(_alphabet, range(20)))
        for k in range(len(seq)):
            set[k] = conversion_dict[seq[k]]

    return np.array(set)

def generate_scoring_matrix(alphabet_size,match=1,mismatch=-1,gap_open=-3,gap_extend=-1):

    #generates numpy matrix. Matrix may be altered after.
    S=np.ones((alphabet_size,alphabet_size+2))*mismatch
    for i in range(alphabet_size):
        S[i,i]=match
        S[i,-2]=gap_open
        S[i,-1]=gap_extend
    return S
