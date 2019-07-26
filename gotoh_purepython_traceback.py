
def zeros(a, b):
    return [[0 for i in range(a)] for j in range(b)]

def ones(a, b, val):
    return [[val for i in range(a)] for j in range(b)]

def identity(a):
    k = zeros(a, a)
    for i in range(k):
        k[i][i] = 1
    return k


def report_alignment(input,mol_type="DNA",alphabet="auto"):
    # print(type(input))
    if type(input)==tuple:
        a,b,s=input
    else:
        print("Alignment score:",input)
        return
    # print(np.unique(a))
    # if mol_type=="auto":
    #     if len(np.unique(a))>6 and len(np.unique(b))>6:
    #         mol_type="Protein"
    #     else:
    #         mol_type="DNA"

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


# noinspection PyPep8Naming
def globalms_align(A, B, match, mismatch, gap_open, gap_extend, penalize_extend_when_opening=False):
    # """ Initializes and fills up the matrices and calculates the alignment score.  """
    n = len(A)
    m = len(B)

    neg_inf = -float("inf")

    # s function
    def s(x, y):
        if x == y:
            return match
        else:
            return mismatch

    # g function

    def g(k):
        if penalize_extend_when_opening:
            return gap_open + gap_extend * k
        else:
            return gap_open + gap_extend * (k - 1)

    # construct and initialize the matrices
    D = zeros(m + 1, n + 1)
    D[0][0] = 0
    D[0][1] = g(1)
    D[1][0] = g(1)
    for i in range(2, n + 1):
        D[i][0] = D[i - 1][0] + gap_extend
    for j in range(2, m + 1):
        D[0][j] = D[0][j - 1] + gap_extend

    P = zeros(m + 1, n + 1)
    for i in range(1, n + 1):
        P[i][0] = neg_inf
    for j in range(1, m + 1):
        P[0][j] = neg_inf

    Q = zeros(m + 1, n + 1)
    for j in range(1, m + 1):
        Q[0][j] = neg_inf
    for i in range(1, n + 1):
        Q[i][0] = neg_inf

    #for traceback
    dir_on_D = zeros(m + 1, n + 1)
    dir_on_P = zeros(m + 1, n + 1)
    dir_on_Q = zeros(m + 1, n + 1)
    for j in range(1, m+1):
        dir_on_D[0][j]=100
        dir_on_Q[0][ j] = 1
    for i in range(1, n + 1):
        dir_on_D[i][0]=10
        dir_on_P[i][ 0] = 1

    # fill up the rest of the matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            P[i][j] = max(D[i - 1][j] + g(1), P[i - 1][j] + gap_extend)
            Q[i][j] = max(D[i][j - 1] + g(1), Q[i][j - 1] + gap_extend)
            D[i][j] = max(D[i - 1][j - 1] + s(A[i - 1], B[j - 1]), P[i][j], Q[i][j])

            dir_on_P[i][ j] = D[i - 1][ j] + g(1) < P[i - 1][ j] + gap_extend
            dir_on_Q[i][ j] = D[i][ j - 1] + g(1) < Q[i][ j - 1] + gap_extend
            if D[i][ j] == D[i - 1][ j - 1] + s(A[i - 1], B[j - 1]):
                dir_on_D[i][ j] = 1
            if D[i][ j] == P[i][ j]:
                dir_on_D[i][ j] = dir_on_D[i][ j] + 10
            if D[i][ j] == Q[i][ j]:
                dir_on_D[i][ j] = dir_on_D[i][ j] + 100

    Score, level = D[n][m], 0
    if P[n][m] > Score:
        Score, level = P[n][m], 0
    if Q[n][m] > Score:
        Score, level = Q[n][m], 0

    al1=[]
    al2=[]
    i,j=n,m

    while i!=0 or j!=0:
        if level==0:
            if (dir_on_D[i][j]//10)%10==1:  level=1
            elif (dir_on_D[i][j]//100)%10==1: level=2
            elif dir_on_D[i][ j] % 10 == 1:
                i,j=i-1,j-1
                al1.append(A[i])
                al2.append(B[j])
        elif level==1:
            if dir_on_P[i][j]==0: level=0
            i=i-1
            al1.append(A[i])
            al2.append(-1)
        elif level==2:
            if dir_on_Q[i][j]==0: level=0
            j=j-1
            al1.append(-1)
            al2.append(B[j])

    return al1[::-1],al2[::-1],Score #orientation corrected arrays and best score


# @jit(nopython=True,cache=True)
def globalds_align(A, B, subs_mat, gap_open=-3, gap_extend=-1, penalize_extend_when_opening=False):
    """ Initializes and fills up the matrices and calculates the alignment score.  """

    n = len(A)
    m = len(B)

    neg_inf = -float("inf")

    # s function
    def s(x, y):
        return subs_mat[x][y]

    # g function
    def g(k):
        if penalize_extend_when_opening:
            return gap_open + gap_extend * k
        else:
            return gap_open + gap_extend * (k - 1)

    # construct and initialize the matrices
    D = zeros(m + 1, n + 1)
    D[0][1] = g(1)
    D[1][0] = g(1)
    for i in range(2, n + 1):
        D[i][0] = D[i - 1][0] + gap_extend
    for j in range(2, m + 1):
        D[0][j] = D[0][j - 1] + gap_extend

    P = zeros(m + 1, n + 1)
    for i in range(1, n + 1):
        P[i][0] = neg_inf
    for j in range(1, m + 1):
        P[0][j] = neg_inf

    Q = zeros(m + 1, n + 1)
    for j in range(1, m + 1):
        Q[0][j] = neg_inf
    for i in range(1, n + 1):
        Q[i][0] = neg_inf

    # for traceback
    dir_on_D = zeros(m + 1, n + 1)
    dir_on_P = zeros(m + 1, n + 1)
    dir_on_Q = zeros(m + 1, n + 1)
    for j in range(1, m + 1):
        dir_on_D[0][j] = 100
        dir_on_Q[0][j] = 1
    for i in range(1, n + 1):
        dir_on_D[i][0] = 10
        dir_on_P[i][0] = 1

    # fill up the rest of the matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            P[i][j] = max(D[i - 1][j] + g(1), P[i - 1][j] + gap_extend)
            Q[i][j] = max(D[i][j - 1] + g(1), Q[i][j - 1] + gap_extend)
            D[i][j] = max(D[i - 1][j - 1] + s(A[i - 1], B[j - 1]), P[i][j], Q[i][j])

            dir_on_P[i][ j] = D[i - 1][ j] + g(1) < P[i - 1][ j] + gap_extend
            dir_on_Q[i][ j] = D[i][ j - 1] + g(1) < Q[i][ j - 1] + gap_extend
            if D[i][ j] == D[i - 1][ j - 1] + s(A[i - 1], B[j - 1]):
                dir_on_D[i][ j] = 1
            if D[i][ j] == P[i][ j]:
                dir_on_D[i][ j] = dir_on_D[i][ j] + 10
            if D[i][ j] == Q[i][ j]:
                dir_on_D[i][ j] = dir_on_D[i][ j] + 100

    Score, level = D[n][m], 0
    if P[n][m] > Score:
        Score, level = P[n][m], 0
    if Q[n][m] > Score:
        Score, level = Q[n][m], 0

    al1=[]
    al2=[]
    i,j=n,m

    while i!=0 or j!=0:
        if level==0:
            if (dir_on_D[i][j]//10)%10==1:  level=1
            elif (dir_on_D[i][j]//100)%10==1: level=2
            elif dir_on_D[i][ j] % 10 == 1:
                i,j=i-1,j-1
                al1.append(A[i])
                al2.append(B[j])
        elif level==1:
            if dir_on_P[i][j]==0: level=0
            i=i-1
            al1.append(A[i])
            al2.append(-1)
        elif level==2:
            if dir_on_Q[i][j]==0: level=0
            j=j-1
            al1.append(-1)
            al2.append(B[j])

    return al1[::-1],al2[::-1],Score #orientation corrected arrays and best score

# @jit(nopython=True,cache=True)
def globaldm(A, B, subs_mat, gap_col=4):
    """localdm uses seperate gap penalties for different residues"""

    n = len(A)
    m = len(B)

    neg_inf = -float("inf")

    # s function
    def s(x, y):
        return subs_mat[x][y]

    # g function
    def g(x):
        return subs_mat[x][gap_col]

    # construct and initialize the matrices
    D = zeros(m + 1, n + 1)

    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + subs_mat[A[i - 1], gap_col]
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + subs_mat[B[j - 1], gap_col]
    # print(D)

    P = zeros(m + 1, n + 1)
    for i in range(1, n + 1):
        P[i][0] = neg_inf
    for j in range(1, m + 1):
        P[0][j] = neg_inf

    Q = zeros(m + 1, n + 1)
    for j in range(1, m + 1):
        Q[0][j] = neg_inf
    for i in range(1, n + 1):
        Q[i][0] = neg_inf

    # fill up the rest of the matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            P[i][j] = max(D[i - 1][j] + g(A[i - 1]), P[i - 1][j] + g(A[i - 1]))
            Q[i][j] = max(D[i][j - 1] + g(B[j - 1]), Q[i][j - 1] + g(B[j - 1]))
            D[i][j] = max(D[i - 1][j - 1] + s(A[i - 1], B[j - 1]), P[i][j], Q[i][j])

    Score, level = D[n][m], 0
    if P[n][m] > Score:
        Score, level = P[n][m], 0
    if Q[n][m] > Score:
        Score, level = Q[n][m], 0
    return Score


# @jit(nopython=True,cache=True)
def globaldd(A, B, subs_mat, gap_open_col=4, gap_extend_col=5, penalize_extend_when_opening=False,
             halve_gap_open=False):
    """ whenever entering or exiting a gap the gap openning penalty is applied.
    If the gap flanks the sequences than only one gap openning penalty would be applied."""
    n = len(A)
    m = len(B)
    neg_inf = -float("inf")

    # this is used so that localdd and locads may use same substitution matrix and fid comparable scores.
    if halve_gap_open:
        for row in range(len(subs_mat)):
            subs_mat[row][gap_open_col] = subs_mat[row][gap_open_col] / 2

    # s function
    def s(x, y):
        return subs_mat[x][y]

    # g functions
    def go(x):
        if penalize_extend_when_opening:
            return subs_mat[x][gap_open_col] + subs_mat[x][gap_extend_col]
        else:
            return subs_mat[x][gap_open_col]

    def ge(x):
        return subs_mat[x][gap_extend_col]

    # construct and initialize the matrices
    D = zeros(m + 1, n + 1)
    D[0][0] = 0
    D[1][0] = subs_mat[A[0]][gap_open_col]
    D[0][1] = subs_mat[B[0]][gap_open_col]
    if penalize_extend_when_opening:
        D[1][0] = D[1][0] + subs_mat[A[0], gap_extend_col]
        D[0][1] = D[0][1] + subs_mat[B[0], gap_extend_col]
    for i in range(2, n + 1):
        D[i][0] = D[i - 1][0] + subs_mat[A[i - 1], gap_extend_col]
    for j in range(2, m + 1):
        D[0][j] = D[0][j - 1] + subs_mat[B[j - 1], gap_extend_col]

    P = zeros(m + 1, n + 1)
    for i in range(1, n + 1):
        P[i][0] = neg_inf
    for j in range(1, m + 1):
        P[0][j] = neg_inf

    Q = zeros(m + 1, n + 1)
    for j in range(1, m + 1):
        Q[0][j] = neg_inf
    for i in range(1, n + 1):
        Q[i][0] = neg_inf

    # fill up the rest of the matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            Qg = go(B[j - 1])
            Pg = go(A[i - 1])
            P[i][j] = max(D[i - 1][j] + go(A[i - 1]), P[i - 1][j] + ge(A[i - 1]))
            Q[i][j] = max(D[i][j - 1] + go(B[j - 1]), Q[i][j - 1] + ge(B[j - 1]))
            D[i][j] = max(D[i - 1][j - 1] + s(A[i - 1], B[j - 1]), P[i][j] + Pg, Q[i][j] + Qg)

    Score, level = D[n][m], 0
    if P[n][m] > Score:
        Score, level = P[n][m], 0
    if Q[n][m] > Score:
        Score, level = Q[n][m], 0
    return Score


# @jit(nopython=True,cache=True)
def localms_align(A, B, match=1, mismatch=-1, gap_open=-3, gap_extend=-1, penalize_extend_when_opening=False):
    """ Initializes and fills up the matrices and calculates the alignment score.  """

    n = len(A)
    m = len(B)

    neg_inf = -float("inf")

    # s function
    def s(x, y):
        if x == y:
            return match
        else:
            return mismatch

    # g function
    def g(k):
        if penalize_extend_when_opening:
            return gap_open + gap_extend * k
        else:
            return gap_open + gap_extend * (k - 1)

    # construct and initialize the matrices
    D = zeros(m + 1, n + 1)

    P = zeros(m + 1, n + 1)
    for i in range(1, n + 1):
        P[i][0] = neg_inf
    for j in range(1, m + 1):
        P[0][j] = neg_inf

    Q = zeros(m + 1, n + 1)
    for j in range(1, m + 1):
        Q[0][j] = neg_inf
    for i in range(1, n + 1):
        Q[i][0] = neg_inf

    # for traceback
    dir_on_D = zeros(m + 1, n + 1)
    dir_on_P = zeros(m + 1, n + 1)
    dir_on_Q = zeros(m + 1, n + 1)
    for j in range(1, m + 1):
        dir_on_D[0][j] = 100
        dir_on_Q[0][j] = 1
    for i in range(1, n + 1):
        dir_on_D[i][0] = 10
        dir_on_P[i][0] = 1

    # fill up the rest of the matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            P[i][j] = max(D[i - 1][j] + g(1), P[i - 1][j] + gap_extend)
            Q[i][j] = max(D[i][j - 1] + g(1), Q[i][j - 1] + gap_extend)
            D[i][j] = max(0, D[i - 1][j - 1] + s(A[i - 1], B[j - 1]), P[i][j], Q[i][j])
            
            #for traceback:
            dir_on_P[i][j]=D[i-1][ j] + g(1)< P[i-1][ j] + gap_extend
            dir_on_Q[i][j]=D[i][ j-1] + g(1)< Q[i][ j-1] + gap_extend
            if D[i][ j] == D[i - 1][ j - 1] + s(A[i - 1], B[j - 1]):
                dir_on_D[i][ j] = 1
            if D[i][ j] == P[i][ j]:
                dir_on_D[i][ j] = dir_on_D[i][ j] + 10
            if D[i][ j] == Q[i][ j]:
                dir_on_D[i][ j] = dir_on_D[i][ j] + 100
            
    # Score calculation
    max_n, max_m = 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if D[i][j] > D[max_n][max_m]:  # and (i == n or j == m):
                max_n, max_m = i, j
    Score = D[max_n][max_m]

    #Extract one of the best alignments
    level=0 #level starts at 0 when local
    al1 = []
    al2 = []
    i, j = max_n, max_m
    while D[i][j]>0:
        if level==0:
            if (dir_on_D[i][j]//10)%10==1:  level=1
            elif (dir_on_D[i][j]//100)%10==1: level=2
            elif dir_on_D[i][ j] % 10 == 1:
                i,j=i-1,j-1
                al1.append(A[i])
                al2.append(B[j])
        elif level==1:
            if dir_on_P[i][j]==0: level=0
            i=i-1
            al1.append(A[i])
            al2.append(-1)
        elif level==2:
            if dir_on_Q[i][j]==0: level=0
            j=j-1
            al1.append(-1)
            al2.append(B[j])
    return al1[::-1], al2[::-1], Score #orientation corrected arrays and best score



# @jit(nopython=True,cache=True)
def localds_align(A, B, subs_mat, gap_open=-3, gap_extend=-1, penalize_extend_when_opening=False):
    """ Initializes and fills up the matrices and calculates the alignment score.  """

    n = len(A)
    m = len(B)

    neg_inf = -float("inf")

    # s function
    def s(x, y):
        return subs_mat[x][y]

    # g function
    def g(k):
        if penalize_extend_when_opening:
            return gap_open + gap_extend * k
        else:
            return gap_open + gap_extend * (k - 1)

    # construct and initialize the matrices
    D = zeros(m + 1, n + 1)

    P = zeros(m + 1, n + 1)
    for i in range(1, n + 1):
        P[i][0] = neg_inf
    for j in range(1, m + 1):
        P[0][j] = neg_inf

    Q = zeros(m + 1, n + 1)
    for j in range(1, m + 1):
        Q[0][j] = neg_inf
    for i in range(1, n + 1):
        Q[i][0] = neg_inf


    # for traceback
    dir_on_D = zeros(m + 1, n + 1)
    dir_on_P = zeros(m + 1, n + 1)
    dir_on_Q = zeros(m + 1, n + 1)
    for j in range(1, m + 1):
        dir_on_D[0][j] = 100
        dir_on_Q[0][j] = 1
    for i in range(1, n + 1):
        dir_on_D[i][0] = 10
        dir_on_P[i][0] = 1

    # fill up the rest of the matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            P[i][j] = max(D[i - 1][j] + g(1), P[i - 1][j] + gap_extend)
            Q[i][j] = max(D[i][j - 1] + g(1), Q[i][j - 1] + gap_extend)
            D[i][j] = max(0, D[i - 1][j - 1] + s(A[i - 1], B[j - 1]), P[i][j], Q[i][j])

            # for traceback:
            dir_on_P[i][j] = D[i - 1][j] + g(1) < P[i - 1][j] + gap_extend
            dir_on_Q[i][j] = D[i][j - 1] + g(1) < Q[i][j - 1] + gap_extend
            if D[i][j] == D[i - 1][j - 1] + s(A[i - 1], B[j - 1]):
                dir_on_D[i][j] = 1
            if D[i][j] == P[i][j]:
                dir_on_D[i][j] = dir_on_D[i][j] + 10
            if D[i][j] == Q[i][j]:
                dir_on_D[i][j] = dir_on_D[i][j] + 100

    # Score calculation
    max_n, max_m = 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if D[i][j] > D[max_n][max_m]:  # and (i == n or j == m):
                max_n, max_m = i, j
    Score = D[max_n][max_m]

    #Extract one of the best alignments
    level=0 #level starts at 0 when local
    al1 = []
    al2 = []
    i, j = max_n, max_m
    while D[i][j]>0:
        if level==0:
            if (dir_on_D[i][j]//10)%10==1:  level=1
            elif (dir_on_D[i][j]//100)%10==1: level=2
            elif dir_on_D[i][ j] % 10 == 1:
                i,j=i-1,j-1
                al1.append(A[i])
                al2.append(B[j])
        elif level==1:
            if dir_on_P[i][j]==0: level=0
            i=i-1
            al1.append(A[i])
            al2.append(-1)
        elif level==2:
            if dir_on_Q[i][j]==0: level=0
            j=j-1
            al1.append(-1)
            al2.append(B[j])
    return al1[::-1], al2[::-1], Score #orientation corrected arrays and best score



# @jit(nopython=True,cache=True)
def localdm(A, B, subs_mat, gap_col=4):
    """localdm uses seperate gap penalties for different residues"""

    n = len(A)
    m = len(B)

    neg_inf = -float("inf")

    # s function
    def s(x, y):
        return subs_mat[x][y]

    # g function
    def g(x):
        return subs_mat[x][gap_col]

    # construct and initialize the matrices
    D = zeros(m + 1, n + 1)

    P = zeros(m + 1, n + 1)
    for i in range(1, n + 1):
        P[i][0] = neg_inf
    for j in range(1, m + 1):
        P[0][j] = neg_inf

    Q = zeros(m + 1, n + 1)
    for j in range(1, m + 1):
        Q[0][j] = neg_inf
    for i in range(1, n + 1):
        Q[i][0] = neg_inf

    # fill up the rest of the matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            P[i][j] = max(D[i - 1][j] + g(A[i - 1]), P[i - 1][j] + g(A[i - 1]))
            Q[i][j] = max(D[i][j - 1] + g(B[j - 1]), Q[i][j - 1] + g(B[j - 1]))
            D[i][j] = max(0, D[i - 1][j - 1] + s(A[i - 1], B[j - 1]), P[i][j], Q[i][j])

    # Score calculation
    max_n, max_m = 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if D[i][j] > D[max_n][max_m]:  # and (i == n or j == m):
                max_n, max_m = i, j
    Score = D[max_n][max_m]
    return Score


# @jit(nopython=True,cache=True)
def localdd(A, B, subs_mat, gap_open_col=4, gap_extend_col=5, penalize_extend_when_opening=False, halve_gap_open=False):
    """localdd uses seperate gap open and gap extenstion penalties for different residues"""

    # this is used so that localdd and locads may use same substitution matrix and fid comparable scores.
    if halve_gap_open:
        for row in range(len(subs_mat)):
            subs_mat[row][gap_open_col] = subs_mat[row][gap_open_col] / 2

    n = len(A)
    m = len(B)

    neg_inf = -float("inf")

    # s function
    def s(x, y):
        return subs_mat[x][y]

    # g functions
    def go(x):
        if penalize_extend_when_opening:
            return subs_mat[x][gap_open_col] + subs_mat[x][gap_extend_col]
        else:
            return subs_mat[x][gap_open_col]

    def ge(x):
        return subs_mat[x][gap_extend_col]

    # construct and initialize the matrices .only the first line is adequate for local alignments
    D = zeros(m + 1, n + 1)

    P = zeros(m + 1, n + 1)
    for i in range(1, n + 1):
        P[i][0] = neg_inf
    for j in range(1, m + 1):
        P[0][j] = neg_inf

    Q = zeros(m + 1, n + 1)
    for j in range(1, m + 1):
        Q[0][j] = neg_inf
    for i in range(1, n + 1):
        Q[i][0] = neg_inf

    # fill up the rest of the matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            Qg = go(B[j - 1])
            Pg = go(A[i - 1])
            P[i][j] = max(D[i - 1][j] + go(A[i - 1]), P[i - 1][j] + ge(A[i - 1]))
            Q[i][j] = max(D[i][j - 1] + go(B[j - 1]), Q[i][j - 1] + ge(B[j - 1]))
            D[i][j] = max(0, D[i - 1][j - 1] + s(A[i - 1], B[j - 1]), P[i][j] + Pg, Q[i][j] + Qg)

    # Score calculation
    max_n, max_m = 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if D[i][j] > D[max_n][max_m]:  # and (i == n or j == m):
                max_n, max_m = i, j
    Score = D[max_n][max_m]
    return Score


def convert_to_numeric(seq, alphabet="auto", mol_type="auto"):
    seq = seq.upper().replace(" ", "").replace("\n", "").strip("*")
    set = [None] * len(seq)

    if mol_type == "auto":
        if all([(x in "ARNDCQEGHILKMFPSTWYVBZX") for x in seq]):
            mol_type = "Protein"
        if all([x in "UAGCN" for x in seq]):
            mol_type = "RNA"
        if all([x in "TAGCN" for x in seq]):
            mol_type = "DNA"

    if mol_type == "DNA":
        _alphabet = "TAGCN"
        if alphabet != "auto": _alphabet = alphabet
        conversion_dict = dict(zip(_alphabet, range(5)))
        for k in range(len(seq)):
            set[k] = conversion_dict[seq[k]]

    elif mol_type == "RNA":
        _alphabet = "UAGCN"
        if alphabet != "auto": _alphabet = alphabet
        conversion_dict = dict(zip(_alphabet, range(5)))
        for k in range(len(seq)):
            set[k] = conversion_dict[seq[k]]

    # T:0,A:1,G:2,C:3,N:4,-1 is reserved for gap
    elif mol_type == "Protein":
        _alphabet = "ARNDCQEGHILKMFPSTWYVBZX"
        if alphabet != "auto": _alphabet = alphabet
        conversion_dict = dict(zip(_alphabet, range(20)))
        for k in range(len(seq)):
            set[k] = conversion_dict[seq[k]]

    return set


def generate_scoring_matrix(alphabet_size, match=1, mismatch=-1, gap_open=-3, gap_extend=-1):
    # generates numpy matrix. Matrix may be altered after.
    S = ones(alphabet_size + 2, alphabet_size, mismatch)
    for i in range(alphabet_size):
        S[i][i] = match
        S[i][-2] = gap_open
        S[i][-1] = gap_extend
    return S


if __name__ == "__main__":
    from Bio import pairwise2 as pw2
    from myPairwise3 import ProcessTimer as pt

    seq1 = "GGGGGGGGGCCTCATCGTACGT"
    seq2 = "GGGGGGGGGATCGTACGT"
    seq1 = "GGGGTTAGAGTGGT"
    seq2 = "AAAAGTTAGAGTGGAAAT"
    # seq1 = "CAATTTTTTGGGGTTAGAGTGCCGTTTTTTGGG"
    # seq2 = "AATTTTTTGGGGTTATGTGAGTGGTTTTTTGGG"
    # seq1 = "GATTAGATCGGATCGTACGT"
    # seq2 = "GATTAGATATCGTACGT"
    # seq1 = "CCCCCCAAAAACCCGGGGGCCCCC"
    # seq2 = "CCCCCCAAAAAGGGGGCCCCC"
    # seq1="CCAATTTGGCC" #NOW CORRECT #finds correct score but draws wrong alignment
    # seq2="CCAACCCTTTCCCGGCC"
    # seq1="AGGTAGGT" #seperate Score and D[n,m] scores. Score is correct
    # seq2="AGGTAGG"
    # seq1="CCCC"
    # seq2="TCTCCC"
    # seq1="CCCTCT"
    # seq2="CCCC"
    # seq1="CCCC" #now correct #wrong alignment but correct best score as Score
    # seq2="CCCTCT"
    # seq1="AGGGT"
    # seq2="AGGT"
    # seq1="AGCGT" #different D[n,m] and Score. Score is correct
    # seq2="AGCG"
    # seq1="ATTTGG"
    # seq2="ATTTGG"
    # seq1 = "AATTTTTTGGGGTTAGAGTGGTTTTTTGGGGT" #different D[n,m] and Score. Score is correct
    # seq2 = "AATTTTTTGGGGTTAGGAGTGGTTTTTTGGG"
    # seq1="ATGCTAGCTCTATAGC" #CORRECT
    # seq2="CTGATCGTCGATGCA"
    # for t in range(10000):

    # for z in range(100000):
    pt1 = pt()
    i = pw2.align.globalms(seq1, seq2, 1, -1, -3, -1,  penalize_extend_when_opening=False)
    pt1.report()
    print("----\n\tpw2 globalms_align", i)
    pt1 = pt()
    p = pw2.align.localms(seq1, seq2, 1, -1, -3, -1, penalize_extend_when_opening=False)
    pt1.report()
    print("----\npw2 localms_align", p)
    #
    seq1 = convert_to_numeric(seq1)
    seq2 = convert_to_numeric(seq2)

    pt1 = pt()
    a = globalms_align(seq1, seq2, 1, -1, -3, -1, penalize_extend_when_opening=False)
    pt1.report()
    print("----\n\tglobalms_align")
    # print(a)
    report_alignment(a)

    sm = generate_scoring_matrix(alphabet_size=4, match=1, mismatch=-1, gap_open=-1.6 + 0.0001,
                                 gap_extend=-1.00 + 0.0000001)
    # sm[3,4]=-1.05
    # sm[1, 4] = -3.000
    # print(sm)
    # pt1=pt()
    b = globalds_align(seq1, seq2, sm, -3, -1, penalize_extend_when_opening=True)
    pt1.report()
    print("----\n\tglobalds_align")
    # print(b)
    report_alignment(b)
    # pt1=pt()
    # c=globaldm(seq1,seq2,sm)
    # pt1.report()
    # print("----\n\tglobaldm")
    # print(c)
    # pt1=pt()
    d = localms_align(seq1, seq2, 1, -1, -3, -1, penalize_extend_when_opening=False)
    pt1.report()
    print("----\nlocalms_align")
    # print(d)
    report_alignment(d)
    # pt1=pt()
    e = localds_align(seq1, seq2, sm, -3, -1, penalize_extend_when_opening=False)
    pt1.report()
    print("----\nlocalds")
    # print(e)
    report_alignment(e)
    # pt1 = pt()
    f = localdm(seq1, seq2, sm)
    pt1.report()
    print("----\nlocaldm")
    print(f)

    # g=globaldd(seq1,seq2,sm,penalize_extend_when_opening=False)
    # print("----\n\tglobaldd")
    # print(g)
    #
    h = localdd(seq1, seq2, sm, penalize_extend_when_opening=False)
    print("----\nlocaldd")
    print(h)
