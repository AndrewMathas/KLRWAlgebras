r"""
    Compute the graded dimension of  e(i)R^Lambda_\alpha e(j),
    where \Lambda is a dominant weight and i,j\in I^\alpha.

The weighted KLRW algebras are a family of diagram algebras indexed by a
(symmetrisable) quiver. The diagrams that generate these algebras are have red,
solid and ghost strings, which can carry (finitely many) dots. The weighted
KLRW algebras are closely related to the KLR algebras, introduced independently
by Khovanov-Lauda and Rouquier.

Compute the graded dimension of  $e(i)R^Lambda_\alpha e(j)$,
where \Lambda is a dominant weight and $i,j\in I^\alpha$.

By Hu-Shi [HuShi21]_, if $i,j\in I^n$ then

    .. math:

        \dim_q e(i)R^\Lambda_\alpha e(j)
          = \sum_{w\in S_{(i,j)}}
              \prod_{t=1}^n[N^\Lambda(w,i,t)]_{i_t} q_{i_i}^{N^\Lambda(i,t)-1}}

    sage: R = RootSystem(['G', 2])
    sage: klr_cyclotomic_dimension(R, [1], [1,2,1,1,2])

    sage: KLRW_Idempotents(['F',4],[2],6)

AUTHORS:

- Andrew Mathas (2023): initial version

# References

.. [HuShu21] J. Hu and L. Shi
    *Cellularity and subdivision of KLR and weighted KLRW algebras
    :arxiv:`2108.05508`

'' [MaTu21] A. Mathas and D. Tubbenhauer
   *Cellularity and subdivision of KLR and weighted KLRW algebras*
   :arxiv:`2111.12949`

'' [MaTu22] A. Mathas and D. Tubbenhauer
   *Cellularity for weighted KLRW algebras of types $B$, $A^{(2)}, $D^{(2)}$*
   J. Lond. Math. Soc. (2) 107 (2023), no. 3, 1002–1044.
   :doi:`10.1112/jlms.12706`
   :arxiv:`2201.01998`

'' [MaTu23] A. Mathas and D. Tubbenhauer
   *Cellularity of KLR and weighted KLRW algebras via crystals*
   :arxiv:`2309.13867 `

    Andrew Mathas
"""

from copy                                  import deepcopy

from sage.combinat.combination             import Combinations
from sage.combinat.root_system.cartan_type import CartanType
from sage.combinat.tuple                   import Tuples
from sage.graphs.graph                     import Graph
from sage.graphs.graph_plot                import GraphPlot
from sage.groups.perm_gps.permgroup_named  import SymmetricGroup
from sage.misc.cachefunc                   import cached_method
from sage.plot.colors                      import Color
from sage.rings.infinity                   import Infinity
from sage.rings.real_mpfr                  import RR
from sage.structure.sage_object            import SageObject
from sage.typeset.ascii_art                import AsciiArt

from decimal import Decimal

def quantum_integer(q,k):
    '''
    Return the quantum integer (q^k-q^{-k})/(q-q^{-1})

        [1] = 1
        [2] = q + q^-1
        [3] = q^2 + 1 + q-2
        [-k] = -[k]

    '''
    if k < 0:
        return -quantum_integer(q, -k)

    return sum( q**(k-2*i-1) for i in range(k))


def klr_cyclotomic_dimension(C, L, bi, bj=None, base=[], verbose=False, cancelling=True, diagrams=False):
    r"""
    This function implements the Hu-Shi formula for the graded dimension of the
    weight space e(i) R^L_n e(j), where i,j \in I^n. Here:

        - `C`  is the Cartan type
        - `L`  is a list specifying the dominant weight
        - `bi` is a weight in I^n
        - `bj` is a weight in I^n, which defaults to `bi`

    If `verbose` is set to `True` then some details of the computation are
    printed as well.

    EXAMPLES::

        sage: klr_cyclotomic_dimension(['D',4],[2], [2,3,4,1])
        ({1: [()]}, 1)
        sage: klr_cyclotomic_dimension(['D',4],[2], [2,3,4,1], [2,3,4,1])
        ({1: [()]}, 1)
        sage: klr_cyclotomic_dimension(['D',4],[2], [2,3,4,1], [2,4,3,1])
        ({1: [(1,2)]}, 1)
        sage: klr_cyclotomic_dimension(['A',2,1],[0], [0,1,2])
        ({(q + 1/q)*q: [()]}, q^2 + 1)
        sage: klr_cyclotomic_dimension(['A',2,1],[0], [0,2,1])
        ({(q + 1/q)*q: [()]}, q^2 + 1)
        sage: klr_cyclotomic_dimension(['A',2,1],[0], [0,1,2], [0,2,1])
        ({q: [(1,2)]}, q)
        sage: klr_cyclotomic_dimension(['A',1],[1,1],[1],[1])
        ({(q + 1/q)*q: [()]}, q^2 + 1)
        sage: klr_cyclotomic_dimension(['A',1],[1,1],[1],[1])
        ({(q + 1/q)*q: [()]}, q^2 + 1)
        sage: klr_cyclotomic_dimension(['B',3],[2], [2,3,3,2,1])
        ({(q^2 + 1/q^2)*(q + 1/q)^2*q^2: [(1,2)]}, (q^4 + 1)*(q^2 + 1)^2/q^2)
    """
    if verbose:
        def vprint(*args): print(' '.join(f'{a}' for a in args))
    else:
        def vprint(*args): pass

    # bj defaults to bi
    if bj is None:
        bj = bi

    if len(bi) != len(bj):
        raise TypeError(f'{bi} and {bj} must have the same length')

    # Convert C to Cartantype
    try:
        C = CartanType(C)
    except TypeError:
        raise TypeError(f'{C} must be a Cartan type')

    # index set for quiver
    I = C.index_set()
    #vprint(f'{I=}')

    # if bi and bj are strings convert them to a list
    if isinstance(bi,str):
        bi = [Integer(i) for i in bi]

    if isinstance(bj,str):
        bj = [Integer(j) for j in bj]

    if isinstance(base,str):
        base = [Integer(j) for j in base]

    if any(i not in I for i in bi):
        raise TypeError(f'{bi} must in I^n for {I=}')

    if any(j not in I for j in bj):
        raise TypeError(f'{bj} must in I^n for {I=}')

    # shorthand for Cartan matrix entries
    Cij = lambda i,j: C.cartan_matrix()[I.index(i), I.index(j)]

    # shorthand for q_di, where is the Cartan symmetriser
    qdi  = lambda i: q**(C.symmetrizer()[i])

    # work in SymmetricGroup(n)
    n = len(bi)

    q = var('q')

    N = Integer(0) # will become \sum_w \prod_t [N^L(w,bi,t)]q^{N^L(1,bi,t)-1}

    Sn = SymmetricGroup(range(n))
    SN = SymmetricGroup(range(n+len(base)))

    # find the subgroup of Sn that maps bi to bj
    generators = []
    for w in Sn:
        if all( bj[w(i)]==bi[i] for i in range(n) ) and not w.is_one():
            generators.append( prod(SN.simple_reflection(i+len(base)) for i in w.reduced_word()) )

    i = 0
    while i < len(base)-1:
        j = i+1
        while j < len(base) and Cij(base[i], base[j])==0:
            j += 1
        if j<len(base) and base[i] == base[j]:
            generators.append( SN((i,j)) )
        i += 1

    bi = base + bi
    bj = base + bj
    n = len(bi)
    Nwt = lambda w,t: L.count(bi[t]) - sum(Cij(bi[t], bi[j]) for j in range(t) if w(j)<w(t))

    vprint(f'Subgroup of permutations = {generators}') #.replace('[', '< ').replace(']',' >'))
    # first calculate NOne[t] = Nwt(1, t) for t in range(n)
    tally = {}
    NOne = [ Nwt(SN.one(), t)-1 for t in range(n)]
    vprint(f'N(1,t)-1:{" ".join(f"{i:>2}" for i in NOne)}')
    #vprint('qdi:   ',' '.join(f'{qdi(bi[t])}' for t in range(n)))
    for w in SN.subgroup(generators):
        # Sym(i,j) = { w\in Sym_n | wi = j}
        if all( bj[w(i)]==bi[i] for i in range(n) ): # if w(bi) == bj
            vprint(f'N(w,t): ',' '.join(f'{Nwt(w,t):>2}' for t in range(n)))
            Nw = prod([quantum_integer( qdi(bi[t]), Nwt(w,t) )*( qdi(bi[t])**(NOne[t]) ) for t in range(n) ])
            if Nw != 0:
                vprint(f'N({w},t): ',' '.join(f'{Nwt(w,t):>2}' for t in range(n)))
                vprint(f'X({w}):', Nw.factor())
                if cancelling and -Nw in tally:
                    tally[-Nw].pop()
                    if tally[-Nw] == []:
                        # tally is sorted by length so we remove a w of longest length
                        del tally[-Nw] 
                else:
                    if Nw not in tally:
                        tally[Nw] = []
                    tally[Nw].append(SN(w))
                    tally[Nw].sort(key=lambda w: w.length())
                N += Nw

    for Nw in tally:
        vprint(f'{Nw}: {", ".join(f"{w}" for w in tally[Nw])}')

    if N != 0:
        N = N.factor()

    if diagrams:
        for c in tally:
            bi = [f'{i}' for i in bi]
            bj = [f'{j}' for j in bj]
            print(f'{c}:','\n')
            print('\n\n'.join(f'{tikz_perm(w,bi, bj)}' for w in tally[c]), '\n')
    return tally, N

def klr_cyclotomic_dimension_tester(C, L, bi, bj=None, base=[], verbose=False, cancelling=True, diagrams=False):
    '''
    Successively move the base into bi and bj to see if we can find any
    permutations.

    '''
    # Convert C to Cartantype
    try:
        C = CartanType(C)
    except TypeError:
        raise TypeError(f'{C} must be a Cartan type')

    # if bi and bj are strings convert them to a list
    if isinstance(bi,str):
        bi = [Integer(i) for i in bi]

    if isinstance(bj,str):
        bj = [Integer(j) for j in bj]

    if isinstance(base,str):
        base = [Integer(j) for j in base]

    while True:
        print(f'{base=}')
        klr_cyclotomic_dimension(C, L, bi, bj, base, verbose, cancelling, diagrams)
        if base == []:
            break
        i = base.pop()
        bi.insert(0,i)
        bj.insert(0,i)


def NonZeroWeightSpaces(C, L, n):
    """
    Return the list of non-zero weight spaces of e(i)R^L_ne(i), together with
    their graded dimension.


    EXAMPLES::

        sage: NonZeroWeightSpaces(['D',4], [1], 4)
        bi=[1, 2, 4, 3]: 1
        bi=[1, 2, 3, 4]: 1
        sage: NonZeroWeightSpaces(['D',4], [2], 4)
        bi=[2, 4, 3, 1]: 1
        bi=[2, 3, 4, 1]: 1
        bi=[2, 3, 1, 2]: 1
        bi=[2, 4, 1, 2]: 1
        bi=[2, 1, 3, 2]: 1
        bi=[2, 4, 3, 2]: 1
        bi=[2, 1, 4, 2]: 1
        bi=[2, 3, 4, 2]: 1
        bi=[2, 4, 1, 3]: 1
        bi=[2, 1, 4, 3]: 1
        bi=[2, 3, 1, 4]: 1
        bi=[2, 1, 3, 4]: 1
        sage: NonZeroWeightSpaces(['D',4], [3], 4)
        bi=[3, 2, 4, 1]: 1
        bi=[3, 2, 1, 4]: 1
        sage: NonZeroWeightSpaces(['D',4], [4], 4)
        bi=[4, 2, 3, 1]: 1
        bi=[4, 2, 1, 3]: 1
    """
    # Convert C to Cartantype
    try:
        C = CartanType(C)
    except TypeError:
        raise TypeError(f'{C} must be a Cartan type')

    # index set for quiver
    I = C.index_set()

    for bi in Tuples(I,n-len(L)):
        dim = klr_cyclotomic_dimension(C, L[:1], L+bi)
        if dim != 0:
            print(f'bi = {L+bi}: {dim}')

# NonZeroWeightSpaces(['F',4], [1], 4)

class Block:
    """
    A container for holding information about blocks of (cyclotomic) KLR
    algebras. It takes as input:
        - a Cartan type `cartan`
        - a dictionary `b`, where `b[i]` gives the multiplicity of `i`
    """
    def __init__(self, cartan=None, Lam=None, block=None):
        usage = 'usage: Block(<CartanType>, <Lambda>) or Block(<block>)'
        if cartan is not None and Lam is not None and block is None:

            try:
                self._cartan = CartanType(cartan)
                self._Lambda = Lam
                self._residue = []
                self._block = {i:0 for i in self._cartan.index_set()}

            except TypeError:
                raise ValueError(usage)

        elif cartan is None and Lam is None and block is not None:
            self._cartan  = block._cartan
            self._Lambda  = block._Lambda
            self._residue = block._residue.copy()
            self._block   = block._block.copy()

        else:
            raise ValueError(usage)

    def add_residue(self, i):
        """
        Add a residue to `self`.
        """
        if i not in self._block:
            raise ValueError(f'{i} is not a vertex of the quiver')
        self._residue.append(i)
        self._block[i] += 1

    def cartan(self):
        return self._cartan

    def defect(self):
        r'''
        TODO: return the defect of `self`, which is the integer

        .. MATH:

                (\Lambda,\alpha) - (\alpha,\alpha)/2

            where `\alpha` is the simple root
        '''
        raise NotImplementedError

    def residue(self):
        return self._residue


    def simple_root(self):
        return sum(R.root_lattice().basis()[i] for i in self._residue)

    def size(self):
        return len(self._residue)

    def __repr__(self):
        return ''.join(f'{i}' for i in self._residue)

    def __str__(self):
        return ' '.join(f'{i}^{self._block[i]}' for i in self._block)


def CheckForRepeatedPositiveRoots(
        cry,
        L,
        depth=Infinity,
        minimal=False,
        minlength=False,
       verbose=False
    ):
    '''
    Check for vertices of the crystal graph that correspond to the same simple
    root, which implies that they live in a non-semisimple block of the
    corresponding cyclotomic KLR algebra. We do this by:
        - for each vertex v of the crystal graph we find a path from the
          highest weight vector to v
        - we add up the labels
    '''
    if verbose:
        def vprint(*args): print(args)
    else:
        def vprint(*args): pass

    if minlength:
        minimal = True

    cry = CartanType(cry)
    R = RootSystem(cry)
    if not isinstance(L, list):
        L = [L]
    wt = sum(R.weight_space().basis()[k] for k in L)
    crystal = crystals.LSPaths(wt)
    G = crystal.digraph(depth=depth)
    vprint('Asking sage to construct the crystal')
    vlam = crystal.highest_weight_vector()
    simples = { vlam: Block(cartan=cry, Lam=wt) }
    blocks = {}
    vertices = [ vlam ]
    next_vertices = []  # vertices with length one more than the current vertex
    while vertices != []:
        v = vertices.pop()
        vprint(f'Looking for edges from {v}')
        for _,w,i in G.edges(vertices=[v]):
            if w not in simples:
                vprint(f' - there a {i}-edge to {w}')
                new_simple = Block(block=simples[v])
                new_simple.add_residue(i)
                simples[ w ] = new_simple
                next_vertices.append(w)

                bstr = f'{new_simple}'
                if bstr not in blocks:
                    blocks[bstr] = []
                blocks[bstr].append(new_simple)
                if minimal and len(blocks[bstr])>1:
                    if minlength:
                        return blocks[bstr][0].size()
                    return blocks[bstr]

        if vertices == []:
            vertices = next_vertices
            next_vertices = []

    if minimal or minlength:
        blocks = {b: blocks[b] for b in blocks if len(blocks[b])>1}
        indices = list(blocks.keys())
        sorted(indices, key=lambda b: blocks[b][0].size())
        if minlength:
            return oo if blocks=={} else blocks[0].size()
        return blocks[ indices[0] ] if blocks!={} else []

    return {b: blocks[b] for b in blocks if len(blocks[b])>1}

def FindMaximalLengthCrystalStrings(cry, i, depth=Infinity, verbose=False):
    '''
    Check for maximal length of an i-string in the crystal graph of the
    fundamental weight `Lambda_i`

    EXAMPLES:

        sage: FindMaximalLengthCrystalStrings(['D',8], 4)
        {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}
    '''
    if verbose:
        def vprint(*args): print(args)
    else:
        def vprint(*args): pass

    cry = CartanType(cry)
    Lambda = RootSystem(cry).weight_space().basis()[i]
    vprint('Constructing the crystal')
    crystal = crystals.LSPaths(Lambda)
    G = crystal.digraph(depth=depth)
    vlam = crystal.highest_weight_vector()
    zeros = {i: 0 for i in cry.index_set()}
    lengths = zeros.copy()
    vertex_lengths = {}
    vertex_lengths[vlam] = zeros.copy()
    vertices = [ vlam ]
    next_vertices = []  # vertices with length one more than the current vertex
    current_depth = 0   # keep track of the current depth in the crystal graph
    while vertices != []:
        v = vertices.pop()
        vprint(f'Looking for edges from {v}')
        for _,w,i in G.edges(vertices=[v]):
            if w not in next_vertices:
                next_vertices.append(w)
                vertex_lengths[w] = zeros.copy()

            vertex_lengths[w][i] = vertex_lengths[v][i] + 1
            lengths[i] = max(lengths[i], vertex_lengths[w][i])
            if lengths[i] == vertex_lengths[w][i]:
                vprint(f'{lengths[i]}: {i=}, {current_depth=}, {w=}')

        if vertices == []:
            current_depth += 1
            if current_depth < depth:
                vertices = next_vertices
            next_vertices = []

    return lengths


class CrystalArm:
    '''
    A circuit inside a crystal graph such that all edges are of colour `i` or colour `j`.
    '''
    def __init__(self, crystal, cartan_matrix, v, colours, vertices=[], word='', ancestor=None):
        self._crystal = crystal
        self._cartan = cartan_matrix
        self._colours = colours
        self._star = ''
        if len(colours) == 2:
            i = cartan_matrix.index_set().index( colours[0] )
            j = cartan_matrix.index_set().index( colours[1] )
            if cartan_matrix[i][j]<0:
                # add a star if the vertices are adjacent in the crystal graph
                self._star = '*'
        self.vertices = vertices+[v]
        self.word = f'{word}'  # word for right arm
        self.ancestor = ancestor

    def __repr__(self):

        return self.word+self._star

    def extensions(self):
        '''
        If possible extend the directed loop along both the left and right hand path
        '''
        extensions = []
        for _,w,i in self._crystal.edges(vertices=[self.vertices[-1]]):
            try:
                ancestor = self.vertices[-2]
            except IndexError:
                ancestor = None

            if i in self._colours:
                extensions.append( CrystalArm(self._crystal, self._cartan, w, self._colours, self.vertices, f'{self.word}{i}') )
        return extensions

    def length(self):
        r'''
        Return the length of the arm.
        '''
        return len(self.vertices)-1

    def __lt__(self, other):
        '''
        Order according to the word
        '''
        return self.word < other.word

    def __eq__(self, other):
        r'''
        Equality test for arms arms.
        '''
        return self.word == other.word

def FindPathsInBlock(
        cry,
        L,
        residues,
        detour_only = True, # return only graphs for detour permutations
        verbose     = False,
    ):
    '''
    Return the subgraph of all paths that have residue sequence that is a
    permutation of `residue`

    EXAMPLES:

        sage: FindPathsInBlock(['A',4],[2], '213')
    '''

    if verbose:
        def vprint(args): print(args)
    else:
        def vprint(args): pass

    cry = CartanType(cry)
    cmat = cry.cartan_matrix()
    if not isinstance(L, list):
        L = [L]
    Lambda = sum(RootSystem(cry).weight_space().basis()[k] for k in L)
    vprint(f'Constructing the {cry}-crystal of weight {Lambda}')
    crystal = crystals.LSPaths(Lambda)

    # keep track of the residues still required in the path
    delta = {i: i for i in cry.index_set()}
    for i in residues:
        delta[int(i)] += 1

    G = crystal.digraph(depth=len(residues))
    depth = 0
    # inductively construct the paths keeping track of the residues we still need
    vlam = crystal.highest_weight_vector()
    paths = [([vlam], delta)]
    while depth < len(residues) and paths != []:
        vprint(f'Number of paths: {len(paths)}')
        depth += 1
        vprint(f'{depth=}')
        new_paths = []
        for path in paths:
            for _,w,i in G.edges(vertices=[path[0][-1]]):
                if path[1][i] > 1:
                    new_path = path[0].copy()
                    new_path.append(w)
                    new_delta = path[1].copy()
                    new_delta[i] -= 1
                    new_paths.append( (new_path, new_delta) )

        paths = new_paths

    vprint(f'{paths=}')
    # construct and return the subgraphs of all detour permutations
    if detour_only:
        detours = {} # will contain the list of all detour permutation subgraphs
        for path in paths:
            edges = path[0]
            sink = edges[-1]
            if sink not in detours:
                detours[sink] = dict(vertices=[vlam], edges=[])
            for v in range(1,len(edges)):
                if edges[v] not in detours[sink]['vertices']:
                    detours[sink]['vertices'].append( edges[v] )
                detours[sink]['edges'].append(
                    ( detours[sink]['vertices'].index(edges[v-1]), detours[sink]['vertices'].index(edges[v]) )
                )
        return [ Graph([range(len(detours[G]['vertices'])), detours[G]['edges']],format='vertices_and_edges') for G in detours ]

    # construct and return the subgraph for this block
    vertices = [ crystal.highest_weight_vector() ]
    edges = []
    for path in paths:
        for i in range(1,len(path[0])):
            if path[0][i] not in vertices:
                vertices.append(path[0][i])
            edges.append( (vertices.index(path[0][i-1]), vertices.index(path[0][i])) )
            vprint('{edges=}')


    return Graph([range(len(vertices)), edges])

def FindMinimalCrystalLoops(
        cry,
        L,
        depth=Infinity,
        verbose=False,
        length=None,
        add_path=False,
        only_adjacent=0,
        collect=True,
        check_plactic=False
    ):
    '''
    Return the list of all minimal length loops in the crystal graph that have
    edges of colours i and j.

    EXAMPLES:

        sage: FindMaximalLengthCrystalStrings(['D',8], 4)
        {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}

        sage: for i in range(1,7):
        ....:     FindMinimalCrystalLoops(['E',6], L=[i],add_path=True, only_adjacent=2, length=2)
        []
        []
        []
        [(13, 45*, 54*, '4315465324243', ''),
        (15, 56*, 65*, '431546532424354', ''),
        (15, 13*, 31*, '431546532424354', ''),
        (16, 56*, 65*, '4315465324243514', ''),
        (16, 13*, 31*, '4315465324243546', ''),
        (16, 13*, 31*, '4315465324243545', ''),
        (17, 13*, 31*, '43154653242435456', ''),
        (17, 13*, 31*, '43154653242435452', ''),
        (18, 34*, 43*, '431256445254334652', ''),
        (18, 45*, 54*, '431542234436554132', ''),
        (18, 13*, 31*, '431546532424354566', ''),
        (19, 13*, 31*, '4315465324243545662', ''),
        (22, 34*, 43*, '4312544536362541435452', ''),
        (22, 24*, 42*, '4312544536362541432453', ''),
        (24, 24*, 42*, '431245345432624156535443', ''),
        (29, 45*, 54*, '43125445363625414324532145436', '')]
        []
        []

    '''

    if verbose:
        def vprint(args): print(args)
    else:
        def vprint(args): pass

    if length is None:
        good_length = lambda loop: True
    else:
        if type(length) != type([]):
            length = [length]
        good_length = lambda loop: loop[1].length() in length

    cry = CartanType(cry)
    cmat = cry.cartan_matrix()
    if not isinstance(L, list):
        L = [L]
    Lambda = sum(RootSystem(cry).weight_space().basis()[k] for k in L)
    vprint(f'Constructing the {cry}-crystal of weight {Lambda}')
    crystal = crystals.LSPaths(Lambda)
    G = crystal.digraph(depth=depth)
    W = CoxeterGroup(cry)
    descents = lambda w: W.prod(W.simple_reflection(int(i)) for i in w).descents()
    vlam = crystal.highest_weight_vector()
    crystal_loops = []  # loops we find in the crystal graph
    vertices = [ (vlam, None) ] # vertices at depth current_depth
    next_vertices = []  # vertices with length one more than the current vertex
    current_depth = 0   # keep track of the current depth in the crystal graph
    vertex_path = {vlam: ''}
    while vertices != []:
        v, ancestor = vertices.pop()
        vprint(f'Looking for edges from {v=} with {ancestor=}')
        edges = [ (w,i) for _,w,i in G.edges(vertices=[v]) ]
        # update the next set of edges
        for x,i in edges:
            vertex_path[x] = f'{vertex_path[v]}{i}'
            if (x,v) not in next_vertices:
                next_vertices.append( (x,v) )
        # find all loops from all possible pairs of edges
        for x,y in Combinations(edges,2):
            colours = [x[1], y[1]]
            xarms = [ CrystalArm(G, cmat, x[0], colours, [v], f'{x[1]}', ancestor) ]
            yarms = [ CrystalArm(G, cmat, y[0], colours, [v], f'{y[1]}', ancestor) ]
            found = False
            while not ( found or xarms == [] or yarms == [] ):
                a = 0
                vprint(f'{xarms=}, {yarms=}')
                while a < len(xarms):
                    b = 0
                    while b < len(yarms):
                        if xarms[a].vertices[-1] == yarms[b].vertices[-1]:
                            # these two arms join in a loop
                            if xarms[a]._star == '*' or only_adjacent==0:
                                hash = ''
                                if only_adjacent:
                                    # check to see if one of the colours is a
                                    # descent of some suffix of the path
                                    i, j = xarms[a]._colours
                                    k = 0
                                    while hash == '' and k < len(vertex_path[v]):
                                        k += 1
                                        des = descents(vertex_path[v][-k:])
                                        if i in des or j in des:
                                            hash = '#'

                                if only_adjacent < 2 or hash=='':
                                    if xarms[a]<yarms[b]:
                                        loop = (current_depth, xarms[a], yarms[b], vertex_path[v], hash) if add_path else (current_depth, xarms[a], yarms[b])
                                    else:
                                        loop = (current_depth, yarms[b], xarms[a], vertex_path[v], hash) if add_path else (current_depth, yarms[b], xarms[a])

                                    if loop not in crystal_loops and good_length(loop):
                                        if only_adjacent > 2:
                                            print(f'Exception found: {loop}')
                                            return
                                        if collect:
                                            crystal_loops.append(loop)
                                        else:
                                            print(loop)

                                # mark that we have found a loop
                                found = True

                        b += 1
                    a += 1

                # extend the xarms that remain - the extensions are unique
                if not found:
                    newx = []
                    for a in xarms:
                        newx.extend( a.extensions() )
                    xarms = newx

                    # extend the yarms that remain - the extensions are unique
                    newy = []
                    for b in yarms:
                        newy.extend( b.extensions() )
                    yarms = newy

        if vertices == []:
            current_depth += 1
            if current_depth < depth:
                vertices = next_vertices
            next_vertices = []

    return crystal_loops

def FindMinimalCrystalSquares(cry, L, depth=Infinity, verbose=False, add_path=False, only_adjacent=0, collect=True):
    '''
    Return the list of all minimal length loops in the crystal graph that have
    edges of colours i and j.

    EXAMPLES:

        sage: FindMaximalLengthCrystalStrings(['D',8], 4)
        {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}

        sage: for i in range(1,7):
        ....:     FindMinimalCrystalLoops(['E',6], L=[i],add_path=True, only_adjacent=2, length=2)
        []
        []
        []
        [(13, 45*, 54*, '4315465324243', ''),
        (15, 56*, 65*, '431546532424354', ''),
        (15, 13*, 31*, '431546532424354', ''),
        (16, 56*, 65*, '4315465324243514', ''),
        (16, 13*, 31*, '4315465324243546', ''),
        (16, 13*, 31*, '4315465324243545', ''),
        (17, 13*, 31*, '43154653242435456', ''),
        (17, 13*, 31*, '43154653242435452', ''),
        (18, 34*, 43*, '431256445254334652', ''),
        (18, 45*, 54*, '431542234436554132', ''),
        (18, 13*, 31*, '431546532424354566', ''),
        (19, 13*, 31*, '4315465324243545662', ''),
        (22, 34*, 43*, '4312544536362541435452', ''),
        (22, 24*, 42*, '4312544536362541432453', ''),
        (24, 24*, 42*, '431245345432624156535443', ''),
        (29, 45*, 54*, '43125445363625414324532145436', '')]
        []
        []

    '''

    if verbose:
        def vprint(args): print(args)
    else:
        def vprint(args): pass

    cry = CartanType(cry)
    cmat = cry.cartan_matrix()
    if not isinstance(L, list):
        L = [L]
    Lambda = sum(RootSystem(cry).weight_space().basis()[k] for k in L)
    vprint(f'Constructing the {cry}-crystal of weight {Lambda}')
    crystal = crystals.LSPaths(Lambda)
    G = crystal.digraph(depth=depth)
    W = CoxeterGroup(cry)
    descents = lambda w: W.prod(W.simple_reflection(int(i)) for i in w).descents()
    vlam = crystal.highest_weight_vector()
    crystal_loops = []          # loops we find in the crystal graph
    vertices = [ (vlam, None) ] # vertices at depth current_depth
    next_vertices = []          # vertices with length one more than the current vertex
    current_depth = 0           # keep track of the current depth in the crystal graph
    vertex_path = {vlam: ''}
    while vertices != []:
        v, ancestor = vertices.pop()
        vprint(f'Looking for edges from {v=} with {ancestor=}')
        edges = [ (w,i) for _,w,i in G.edges(vertices=[v]) ]
        # update the next set of edges
        for x,i in edges:
            vertex_path[x] = f'{vertex_path[v]}{i}'
            if (x,v) not in next_vertices:
                next_vertices.append( (x,v) )
        # find all loops from all possible pairs of edges
        for x,y in Combinations(edges,2):
            colours = [x[1], y[1]]
            xarms = [ CrystalArm(G, cmat, x[0], colours, [v], f'{x[1]}', ancestor) ]
            yarms = [ CrystalArm(G, cmat, y[0], colours, [v], f'{y[1]}', ancestor) ]
            found = False
            while not ( found or xarms == [] or yarms == [] ):
                a = 0
                vprint(f'{xarms=}, {yarms=}')
                while a < len(xarms):
                    b = 0
                    while b < len(yarms):
                        if xarms[a].vertices[-1] == yarms[b].vertices[-1]:
                            # these two arms join in a loop
                            if xarms[a]._star == '*' or only_adjacent==0:
                                hash = ''
                                if only_adjacent:
                                    # check to see if one of the colours is a
                                    # decent of some suffix of the path
                                    i, j = xarms[a]._colours
                                    k = 0
                                    while hash == '' and k < len(vertex_path[v]):
                                        k += 1
                                        des = descents(vertex_path[v][-k:])
                                        if i in des or j in des:
                                            hash = '#'

                                if only_adjacent < 2 or hash=='':
                                    if xarms[a]<yarms[b]:
                                        loop = (current_depth, xarms[a], yarms[b], vertex_path[v], hash) if add_path else (current_depth, xarms[a], yarms[b])
                                    else:
                                        loop = (current_depth, yarms[b], xarms[a], vertex_path[v], hash) if add_path else (current_depth, yarms[b], xarms[a])

                                    if loop not in crystal_loops and good_length(loop):
                                        if only_adjacent > 2:
                                            print(f'Exception found: {loop}')
                                            return
                                        if collect:
                                            crystal_loops.append(loop)
                                        else:
                                            print(loop)

                                # mark that we have found a loop
                                found = True

                        b += 1
                    a += 1

                # extend the xarms that remain - the extensions are unique
                if not found:
                    newx = []
                    for a in xarms:
                        newx.extend( a.extensions() )
                    xarms = newx

                    # extend the yarms that remain - the extensions are unique
                    newy = []
                    for b in yarms:
                        newy.extend( b.extensions() )
                    yarms = newy

        if vertices == []:
            current_depth += 1
            if current_depth < depth:
                vertices = next_vertices
            next_vertices = []

    return crystal_loops

def FindMaximalLengthCrystalLoopsAsPerms(cry, L, depth=Infinity, verbose=False, length=None, cancelling=True, base_length=0):
    '''
    Print TikZ code for the permutations contributions to the minimal paths
    '''
    paths = FindMinimalCrystalLoops(cry, L, depth, verbose, length, add_path=True)
    for p in paths:
        dims = klr_cyclotomic_dimension(
            cry,
            [L],
            p[3][-base_length:]+p[1].word,
            p[3][-base_length:]+p[2].word,
            base=p[3][:-base_length],
            cancelling=cancelling
        )
        for d,w in dims[0].items():
            # print out only those that contribute to surviving terms
            if not cancelling or d.coefficients()[-1][0] > 0:
                if -d not in dims[0] or len(dims[0][-d]) != len(w):
                    print('\n', f'{p}: ${d}$', '\n\n'+'\n\n'.join(tikz_perm(v, list(p[3]+p[1].word), list(p[3]+p[2].word)) for v in w))

def CrystalGraph(cry, i, **args):
    """
    Return latex code for the given fundamental crystal
    """
    cry = CartanType(cry)
    Lam = RootSystem(cry).weight_space().basis()
    try:
        LS = crystals.LSPaths(sum(Lam[j] for j in i))
    except TypeError:
        LS = crystals.LSPaths(Lam[i])
    return LS.digraph(subset=LS.subcrystal(**args))

def CrystalGraphPBW(cry, max_depth=3, **args):
    """
    Return latex code for the given fundamental crystal
    """
    B = crystals.infinity.PBW(cry)
    if 'w0' in args:
        B.set_default_long_word(args['w0'])
        del args['w0']
    STP = B.subcrystal(max_depth=max_depth, **args)
    return B.digraph(subset=STP)

def CrystalGraphPBWFundamental(cry, Lam=[1], w0=None, **args):
    B = crystals.infinity.PBW(cry)
    b = B.highest_weight_vector()
    if w0 is not None:
        B.set_default_long_word(w0)

    T = crystals.elementary.T(sum(B.Lambda()[i] for i in Lam))
    t = T[0]
    C = crystals.elementary.Component(cry)
    c = C[0]
    TP = crystals.TensorProduct(C,T,B)
    t0 = TP(c,t,b)
    STP = TP.subcrystal(generators=[t0], **args)
    return TP.digraph(subset=STP)


#####################################################################

__KLRW_DIAGRAM_TIKZ__ = r'''\usepackage{tikz}
\tikzset{
  anchorbase/.style={baseline={([yshift=#1]current bounding box.center)}},
  anchorbase/.default={-0.5ex},
  dot colour/.initial=black,
  dot colour/.default=black,
  redstring/.style = {
     draw=#1!50,fill=none,line width=0.35mm,preaction={draw=#1,line width=2.5pt,-},
     nodes={color=#1}
  },
  redstring/.default={red},
  affine/.style= {redstring={orange}},
  solid/.style = {draw=blue,fill=none,dot colour=blue,line width=0.4mm,nodes={color=blue}},
  ghost/.style = {draw=gray,dashed, fill=none,dot colour=darkgray},
}
'''

__latex_file__ = r'''\documentclass{{article}}

{preamble}

\begin{{document}}

  {body}

\end{{document}}
'''


__KLRW_DIAGRAM__ = r'''
\begin{{tikzpicture}}[scale=2]
  {strings}
\end{{tikzpicture}}
'''


class KLRWString(SageObject):
    '''
    Create a new KLRW string of type affine/ghost/red/solid, residue `i` and
    x-coordinate `xcoord`.

    INPUTS::

        - `type`:         The type of the string: 'affine', 'ghost', 'red' or 'solid'
        - `residue`:      The residue of the string, which is a vertex of the quiver
        - `xcoord`:       The x-coordinate of the string
        - `dots`:         The number of dots on the string
        - `delta`:        The "offset" in the x-coordinate of the string
        - `ghost_degree`: The number of ghost edges (default: None)
        - `anchor`:       The index of the (affine) red string that this string is anchored on

    EXAMPLES::

        sage: path = [[4, 5], [4, 5], [3], [3], [2, 3, 4], [2, 3, 4], [1, 2], [1, 2]]
        sage: KLRWIdempotentDiagram(['A',6], [4,4,3,3,2,2,1,1], path)
    '''

    string_repr = {
        'affine': { 'term': '\033[48;5;208m|\033[0m', 'ascii': 'A'},
        'ghost':  { 'term': '\033[38;5;246m|\033[0m', 'ascii': 'G'},
        'red':    { 'term': '\033[48;5;196m|\033[0m', 'ascii': 'R'},
        'solid':  { 'term': '\033[38;5;27m|\033[0m',  'ascii': 'S'},
    }

    def __init__(self, type, residue, xcoord, dots=0, delta=Decimal('0'), ghost_degree=None, anchor=0):
        self.type = type
        self.residue = residue
        self.xcoord = xcoord - delta
        self._raw_x = xcoord
        self.dots = dots
        self.anchor = anchor

        print(f'ADDING: {type=}, {residue=}, {xcoord=}, {dots=}, {delta=}, {ghost_degree}, {anchor=}')
        # ghosts can have degree 1, 2 or 3
        if self.type == 'ghost':
            self._ghost_style = 'ghost'
            if ghost_degree == 2:
                self._ghost_style += ',double'
            elif ghost_degree == 3:
                self._ghost_style += ',triple'

    def is_ghost(self):
        return self.type == 'ghost'

    def is_solid(self):
        return self.type == 'solid'

    def is_red(self):
        return self.type == 'red'

    def __dotstr__(self):
        if self.dots == 0:
            return f' {self.string_repr[self.type]["term"]}'
        dotted = self.string_repr[self.type]['term'].replace('|','o')
        if self.dots == 1:
            return f' {dotted}'
        return f'{self.dots}{dotted}'

    def __str__(self):
        return self.string_repr[self.type]['term']

    def _ascii_art_(self):
        return self.string_repr[self.type]['ascii']

    def __repr__(self):
        return f'({self.residue}, {self.type}, {self.xcoord})'

    def _latex_(self):
        return getattr(self, f'_latex_{self.type}')()

    def _latex_red(self):
        return r'\draw[redstring]'+f'({self.xcoord},0)node[below]{{ ${self.residue}$ }}--+(0,1);'

    def _latex_affine(self):
        return r'\draw[affine]'+f'({self.xcoord},0)node[below]{{ ${self.residue}$ }}--+(0,1);'

    def _latex_solid(self):
        return r'\draw[solid]'+f'({self.xcoord},0)node[below]{{ ${self.residue}$ }}--+(0,1);'

    def _latex_ghost(self):
        return r'\draw['+f'{self._ghost_style}]({self.xcoord},0)--+(0,1)node[above]{{ ${self.residue}$ }};'

    def __eq__(self, other):
        return self.xcoord == other.xcoord

    def __lt__(self, other):
        return self.xcoord < other.xcoord



def oriented_edges_cartan_type(cart):
    '''
    Return the oriented edges of a Dynkin diagram following the conventions of
    [MaTu23]_
    '''
    # Navigating the internal representation of Cartan types in sage is a
    # nightmare. Fortunately, we can navigate this using the compact form of
    # the string representation of Cartan types.
    cart = CartanType(cart)._repr_(compact=True)
    if '^' in cart:
        # affine quivers
        e = int( cart[1:-2] )
        match cart[0]:
            case 'A':
                if cart.endswith('^1'):
                    # A^{(1)}_e
                    return [(i,i+1,1) for i in range(0,e)]+[(e,0,1)]

                if is_even(e):
                    # A^{(2)}_{2e}
                    e = int(e/2)
                    return [(0,1,2)] + [(i,i+1,1) for i in range(1,e-1)] + [(e-1, e, 2)]

            case 'C':
                return [(0,1,2)] + [(i,i+1,1) for i in range(1,e-1)] + [(e, e-1, 2)]

            case 'D':
                if cart.endswith('^2'):
                    return [(1,0,2)] + [(i,i+1,1) for i in range(1,e-1)] + [(e-1, e, 2)]


    else:
        e = int( cart[1:] )
        match cart[0]:
            case 'A':
                return [(i,i+1,1) for i in range(1,e)]

            case 'B':
                return [(i,i+1,1) for i in range(1,e-1)] + [(e-1, e, 2)]

            case 'C':
                return [(1,2,2)] + [(i,i+1,1) for i in range(2,e)]

            case 'D':
                return [(i,i+1,1) for i in range(1,e-1)] + [(e-2,e,1)]

            case 'E':
                return [(1,2,1), (2,3,1), (3,4,1), (3,5,1)] + [(i,i+1,1) for i in range(5,e)]

            case 'F':
                return [(1,2,1), (2,3,2), (3,4,1)]

            case 'G':
                return [(1,2,3)]

    # If we are still here then the quiver is not supported.
    raise ValueError(f'Quiver of type {cart} are not supported')


class KLRWIdempotentDiagram(SageObject):
    r'''
    Draw a KLRW idempotent diagram given its highest weight ( the red strings),
    the residues of the solid strings.

    INPUTS::

        - `cartan`:  The Cartan type
        - `wt`:      A list specifying the dominant weight, or red strings
        - `path`:    A residue sequence for a level 1 path, or an $\ell$-tuple of residues
        - `vertex`:  The name of the sink vertex in the crystal graph (used only in LaTeXing)
        - `shift`:   The ghost shift (default: `1`)
        - `epsilon`: Sets the string separation when LaTeXing diagrams (default `0.08`)
        - `relabel': Set to `True` to use the Dynkin diagram conventions of [MT]_ (default: `False`)
        - `debug`:   Set to `True` to enable debugging (default: `False`)


    The `path` 
    '''

    # maps from sage's labelling to the labelling of MT
    _cartan_relabelling = {
        # Type C_e labelling: 1 == 2 -- 3 -- ... -- e
        'C': lambda cart, i: cart.rank() - i + 1,
        # Type E_e labelling:
        #             4
        #             |
        #   1 -- 2 -- 3 -- 5 -- ... --- e
        'E': lambda cart, i: i if i not in [2,3,4] else 2 if i==4 else i+1
    }

    # maps from the labelling of MT to sage's labelling
    _relabelling_cartan = {
        'C': lambda cart, i: cart.rank() - i + 1,
        'E': lambda cart, i: i if i not in [2,3,4] else 4 if i==2 else i-1
    }

    def __init__(self, cartan, wt, path, vertex=None, shift=Decimal('1'), epsilon=Decimal('0.08'), relabel=False, debug=False):
        r"""
        Inductively construct the KLRW diagram associated to the path
        `path=(i_1,...,i_n)` in the crystal graph of highest weight `wt` for the
        Cartan type `cartan`.

          - `cartan` the Cartan type such as ['D',4]
          - 'wt'     an ordered list specifying a dominant weight for the  Cartan

          If there are n-strings then red_shift = n+1 and:

            - the red and affine strings have x-coordinates of k*shift, for k\ge1
            - the kth solid its ghost have x-coordinates congruent to -k mod shift,
              with the ghost string being shift coordinates to the right of the
              solid


        EXAMPLES:

            sage: KLRWIdempotentDiagram(['A',5],[3], [3,2,1,4,3,2,5,4,3])
            1        2     2        3     3     3  4     4
            <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-48;5;196m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>
            <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>o<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>o<CSI-0m>  <CSI-38;5;246m>o<CSI-0m> 2<CSI-38;5;27m>o<CSI-0m>  <CSI-48;5;196m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>o<CSI-0m>  <CSI-38;5;27m>o<CSI-0m> 2<CSI-38;5;246m>o<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>o<CSI-0m>
            <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-48;5;196m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>  <CSI-38;5;27m>|<CSI-0m>  <CSI-38;5;246m>|<CSI-0m>
            1  2     2  3     3     3  3     4     4        5   

        """
        self._cartan_type = CartanType(cartan)
        self._weight_space = self._cartan_type.root_system().weight_space()

        # temporarily change the Cartan notation to Kac so that we can extract
        # the data that we need and then restore it to its original format
        notation = str(self._cartan_type.options.notation)
        self._cartan_type.options.notation('Kac')
        cart = self._cartan_type._repr_(compact=True)
        self._cartan_type.options.notation(notation)

        self._letter = cart[0]
        self._rank = int(cart[1:-2]) if '^' in cart else int(cart[1:])
        self._cartan_matrix = self._cartan_type.cartan_matrix()
        self._cartan_I = self._cartan_type.index_set()
        self._epsilon = epsilon
        self._vertex = vertex
        self._wt = wt

        #
        if relabel:
            self._wt = [ self._lebaler(i) for i in self._wt]
            convert = self._lebaler
        else:
            convert = lambda i: i

        if isinstance(path, str):
            self._path = [[convert(int(i)) for i in path]]
        elif isinstance(path, list):
            if path == []:
                self._path = []
            elif isinstance(path[0], int): # assume [0,1,2]
                self._path = [[convert(i) for i in path]]
            elif isinstance(path[0], str): # assume ['012', '231', ...]
                self._path = [ [convert(int(i)) for i in p] for p in path]
            elif isinstance(path[0], list): # assume [[0,1,2],[2,3,1], ...]
                self._path = [ [convert(i) for i in p] for p in path]

        if not hasattr(self, '_path'):
            raise TypeError(f'unrecognised path specification: {path=}')

        # keep track of whether we have added to the LaTeX preamble
        self._have_added_latex_preamble = False

        # pre-compute the ghost degrees = number of edges with tail i
        self._ghost_degree = {i:0 for i in self._cartan_I}
        for edge in oriented_edges_cartan_type(self._cartan_type):
            self._ghost_degree[ edge[0] ] += 1

        # debugging
        if debug:
            self.debug = lambda *args: print(*args)
        else:
            self.debug = lambda *args: None

        # the minimum and maximum string positions
        self._min = None
        self._max = None
        self._solid = Decimal('0')  # records the number of solid strings

        self._ghost_shift = Decimal('1')     # the ghost shift
        self._red_shift = Decimal(f'{len(self._path)+1}') # the shift between affine red strings

        # This list will contain the strings in the diagram
        # Here and below, all strings are of type KLRWString
        self.strings = {}

        # This list will contain the red strings in the diagram
        self._red_strings = []
        for k,i in enumerate(self._wt):
            self.place_string('red', i, Decimal(f'{k}')*self._red_shift, k)
            self._red_strings.append( Decimal(f'{k}')*self._red_shift )

        # This list will contain the affine strings in the diagram
        self._affine_strings = []

        # now add the strings in the path to the diagram
        for red in range(len(self._path)):
            for i in self._path[red]:
                self._solid += Decimal('1')  # increment the number of solid strings
                self.add_string(i, red)      # add a new solid i-string to component `red`

        # and finally add any necessary dots to the solid and ghost strings
        last_i = -1
        dots = 0
        for x in  sorted(self.strings):
            if self.strings[x].type in ['solid']:
                if self.strings[x].residue == last_i:
                    dots += 1
                    self.strings[x].dots = dots
                    if self._ghost_degree[last_i] > 0:
                        self.strings[x+Decimal('1')].dots = dots

                else:
                    dots = 0
                last_i = self.strings[x].residue


    def add_string(self, i, red):
        """
        Add a solid string of residue `i`, together with any ghost strinngs, to
        the diagram by putting the string on the left-hand side of the `red`th
        red string and then dragging it as far as possible to the right.

        ALGORITHM:

         - positions is a list of the strings that are already in the diagram ,
           sorted by their x-coordinates
         - we are placing the mth solid string, and its ghosts into the diagram
           and s and g give the indices of the strings in positions 
        """

        # placing_i is true until the string gets blocked
        placing_i = True

        # record the positions of the existing strings and, initially, put both
        # the solid and ghost strings behind the first string
        positions = sorted(self.strings)
        positions.append( Decimal(f'{positions[-1]}')+self._red_shift*Decimal('10'))
        self.debug('\n'+f'Placing the {self._solid}th solid {i}-string.  {positions=}')

        # initially the solid and ghost strings are left of the red string `red`
        g = 0
        while g<len(positions) and self.strings[positions[g]].anchor < red:
            g += 1
        s = g

        while placing_i and s < len(positions)-1:
            # residues, types and Cartan entries of the adjacent strings
            si = self.strings[ positions[s] ].residue # residue of string next to solid string
            cis = self._cij(i, si)                    # Cartan entry for string next to solid
            stype = self.strings[ positions[s] ].type # type of string next to solid string

            solid_blocked = (stype in ['affine', 'red', 'solid'] and si == i ) or (stype=='ghost' and i>si and cis<0)

            if g < len(positions)-1:
                gi = self.strings[ positions[g] ].residue # residue of string next to ghost string
                cig = self._cij(i, gi)                    # Cartan entry for string next to solid
                gtype = self.strings[ positions[g] ].type # type of string next to ghost string
                ghost_blocked = self._ghost_degree[i] > 0 and gtype == 'solid' and i < gi and cig < 0
            else:
                ghost_blocked = False

            if s<g and (self._ghost_degree[i] == 0 or positions[s]-self._ghost_shift < positions[g]):
                # solid is adjacent to an existing string
                self.debug(f' - solid adjacent: {s=}, {g=}')

                if solid_blocked:
                    # place new solid string here
                    self.debug(f' -> place solid string before #{s} with anchor={self.strings[positions[s]].anchor}')
                    self.place_string('solid', i, self.strings[positions[s]]._raw_x, self.strings[positions[s]].anchor )
                    placing_i = False

                else:
                    s += 1

            elif positions[s]-self._ghost_shift == positions[g]:
                # solid and ghost are both adjacent to existing strings
                self.debug(f' - solid and ghost adjacent: {s=}, {g=}')

                if solid_blocked or ghost_blocked:
                    # place new solid and ghost strings here
                    self.debug(f' ! stopped with {i=}, {s=}, {g=} and {cis=}, {cig=}, s-anchor={self.strings[positions[s]].anchor}')
                    self.place_string('solid', i, self.strings[positions[s]]._raw_x, self.strings[positions[s]].anchor )
                    placing_i = False

                else:
                    g += 1
                    s += 1

            else:
                # only the ghost string is adjacent to an existing string
                self.debug(f' - ghost adjacent: {s=}, {g=}')

                if ghost_blocked:
                    self.debug(f' ! stopped with {i=}, {s=}, {g=} and {cis=}, {cig=}')
                    self.place_string('solid', i, self.strings[positions[g]]._raw_x - self._ghost_shift, self.strings[positions[g]].anchor )
                    placing_i = False

                else:
                    g += 1

        if placing_i:
            # if i has not yet been placed then it sits on an affine string
            self.add_affine_string(i)
            self.place_string('solid', i, self._affine_strings[-1], len(self._red_strings)+len(self._affine_strings))


    def add_affine_string(self, residue):
        """
        Add an affine string of residue `i`

        We ASSUME that the affine string being added is the rightmost string
        the diagram
        """
        xcoord = self._red_shift*len(self._red_strings+self._affine_strings)
        self._affine_strings.append( xcoord )
        self.place_string('affine', residue,xcoord, anchor=len(self._red_strings)+len(self._affine_strings))


    def place_string(self, type, residue, xcoord, anchor):
        """
        Add the specified string to the diagram before string `s`
        """
        self.debug(f'Adding a new {residue}-{type} string with {xcoord=} and {anchor=}')

        # solid and ghost strings are shifted by epsilon times the string number
        if type == 'solid':
            delta = self._solid*self._epsilon
            s = KLRWString('solid', residue, xcoord, delta=delta, anchor=anchor)
            self.strings[s.xcoord] = s

            if self._ghost_degree[residue] > 0:
                s = KLRWString('ghost',
                        residue      = residue,
                        xcoord       = xcoord+self._ghost_shift,
                        delta        = delta,
                        ghost_degree = self._ghost_degree[residue],
                        anchor       = anchor
                )
                self.strings[s.xcoord] = s

        else:
            self.strings[xcoord] = KLRWString(type, residue, xcoord, anchor=anchor)

    def _str(self, i):
        """
        A shorthand for referring to string i
        """
        return self.strings[i]

    def _lebaler(self, i):
        '''
        Relabel the vertex of the quiver to use the MT labelling
        '''
        if self._cartan_type.is_finite() and self._letter in self._relabelling_cartan:
            return  self._relabelling_cartan[self._letter](self._cartan_type, i)

        return i

    @cached_method
    def _relabel(self, i):
        '''
        Relabel the vertex of the quiver to use the MT labelling
        '''
        if self._cartan_type.is_finite() and self._letter in self._cartan_relabelling:
            return  self._cartan_relabelling[self._letter](self._cartan_type, i)

        return i

    def _cij(self, i, j):
        """
        Given `i` and `j` in the index set for the quiver return the
        corresponding Cartan matrix entry
        """
        i = self._relabel(i)
        j = self._relabel(j)

        return self._cartan_matrix[self._cartan_I.index(i), self._cartan_I.index(j)]


    def _latex_(self, labelled=False, preamble=True):
        """
        Return TikZ code for the weight KLRW diagram for the path.

        The LaTeX code requires the klrw_diagram package.

        EXAMPLES:

            sage: d = KLRWIdempotentDiagram(['A', 3], [0], [0,0,1,2])
            sage: latex(d)
        """
        global __KLRW_DIAGRAM_TIKZ__, __KLRW_DIAGRAM__

        if not self._have_added_latex_preamble:
            latex.add_to_preamble(__KLRW_DIAGRAM_TIKZ__)
            self._have_added_latex_preamble = True

        start = ''
        strings = '\n  '.join(f'{latex(self.strings[s])}' for s in self.strings)

        if labelled:
            path = ''.join(f'{i}' for i in self._path)
            if self._vertex is not None:
                path += f', ${latex(self._vertex)}$'
            strings += '\n  ' + r'\node at (0,-0.8) {'+path+'};'

        return start + __KLRW_DIAGRAM__.format(strings=strings)

    def show(self):
        """
        Return a graph that displays the KLRW idempotent diagram `self`
        """
        colours = {
            'affine': Color('orange'),
            'ghost':  Color('grey'),
            'red':    Color('red'),
            'solid':  Color('blue'),
        }
        edges = []
        edge_colours = { colours[c]: [] for c in colours }
        positions = {}
        vertices = {}
        edge = 0
        for s in self.strings:
            st = self.strings[s]

            # attach an edge to the string st and assign it a colour
            edges.append( (edge, edge+1) )
            edge_colours[ colours[st.type] ].append( (edge, edge+1) )

            # vertex labels
            vertices[edge]   = st.residue if st.type != 'ghost' else ''
            vertices[edge+1] = st.residue if st.type != 'solid' else ''

            # vertex positions
            if st.type in ['red', 'affine']:
                positions[edge] = (s,-0.2)
                positions[edge+1] = (s,1.2)
            else:
                positions[edge] = (s,0)
                positions[edge+1] = (s,1)

            # increment the edge counter
            edge += 2

        g = Graph(edges, pos=positions, name="KLRW idempotent")
        GraphPlot(g, {'edge_colors': edge_colours, 'vertex_labels': vertices}).show()


    def __repr__(self):
        """
        Return a coloured string representation of the KLRW idempotent diagram self
        """
        # shorthands for referring to the strings
        strings = sorted(self.strings)  # ordered strings
        S = lambda s: self.strings[s]
        D = lambda s: self.strings[s].__dotstr__()

        top = '  '+'  '.join(f'{S(s).residue if S(s).type == "ghost" else " "}' for s in strings)

        # add the strings and dots
        line = ''.join(f'  {S(s)}' for s in strings) + '\n'
        dots = ''.join(f' {D(s)}' for s in strings) + '\n'

        # add non-ghost residues
        bot = '  '+'  '.join(f'{S(s).residue if S(s).type != "ghost" else " "}' for s in strings)

        return top+'\n'+line+dots+line+bot


    def _display(self):
        """
        Return a coloured string representation of the KLRW idempotent diagram self
        """
        # shorthands for referring to the strings
        strings = sorted(self.strings)  # ordered strings
        S = lambda s: self.strings[s]

        top = '  '+'  '.join(f'{S(s).residue if S(s).type == "ghost" else " "}' for s in strings)

        # add the strings
        line = '  '+'  '.join(f'{display(S(s))}' for s in strings)

        # add non-ghost residues
        bot = '  '+'  '.join(f'{S(s).residue if S(s).type != "ghost" else " "}' for s in strings)

        return top+'\n'+(line+'\n')*2+bot


    def dynkin_diagram():
        """
        Return the Dynkin diagram of the underlying quiver
        """
        return self._cartan_type.dynkin_diagram()

    def cartan_type():
        """
        Return the Cartan type of the underlying quiver
        """
        return self._cartan_type

    def residue(self):
        '''
        Return the residue sequence obtained by reading the residues of the
        solid strings from left to right.
        '''
        coords = sorted(self.strings)
        return [ self.strings[i].residue for i in coords if self.strings[i].type=='solid' ]

    def normal_form(self):
        '''
        Keep applying the monid relations iij = iji until none remain.
        '''
        nf = deepcopy(self)
        coords = sorted(self.strings)
        relating = True # still applying the monoid relations
        while relating:
            r = 0            # currently checking relations at position r
            relating = False # set to True if we apply a relation
            while r<len(coords)-2:
                dr = coords[r]
                print(f'{r=}, {dr=}')
                if nf.strings[dr].type not in ['red', 'affine']:
                    ds = coords[r+1]
                    dt = coords[r+2]
                    i = nf.strings[dr].residue
                    j = nf.strings[dt].residue
                    if nf.strings[ds].residue == i and self._cij(i,j) < 0:
                        if nf.strings[dr].type == 'ghost' and nf.strings[ds].type == 'ghost' and nf.strings[dt].type == 'solid':
                            nf.strings[dr] = KLRWString('solid', i, dr)
                            nf.strings[ds] = KLRWString('ghost', j, ds)
                            nf.strings[dt] = KLRWString('solid', i, dt)
                            relating = True
                            print(f'swapping {r=}: ggs {i}{i}{j}')

                        elif  nf.strings[dr].type == 'solid' and nf.strings[ds].type == 'ghost' and nf.strings[dt].type == 'ghost':
                            nf.strings[dr] = KLRWString('ghost', i, dr)
                            nf.strings[ds] = KLRWString('solid', j, ds)
                            nf.strings[dt] = KLRWString('ghost', i, dt)
                            relating = True
                            print(f'swapping {r=}: sgg {i}{i}{j}')

                # increment
                r += 1

        return nf


def KLRW_Idempotents(cart, L, n, latex=False, verbose=False):
    r'''
    Return a list of the KLRW idempotents indexed by their paths

    EXAMPLES:

        sage: KLRW_Idempotents(['B',3],[2],2)
        [           2
           |  |  |  |
           |  |  |  |
           |  |  |  |
           2  2  3   ,
              1        2
           |  |  |  |  |
           |  |  |  |  |
           |  |  |  |  |
           1     2  2   ]
        sage: [latex(d) for d in KLRW_Idempotents(['B',3],[2],2)]
[
 \begin{tikzpicture}[scale=2]
   \draw[redstring](0,0)node[below]{ $2$ }--+(0,1);
   \draw[solid](-0.08,0)node[below]{ $2$ }--+(0,1);
   \draw[ghost](0.92,0)--+(0,1)node[above]{ $2$ };
   \draw[solid](-1.16,0)node[below]{ $1$ }--+(0,1);
   \draw[ghost](-0.16,0)--+(0,1)node[above]{ $1$ };
 \end{tikzpicture}
 ,

 \begin{tikzpicture}[scale=2]
   \draw[redstring](0,0)node[below]{ $2$ }--+(0,1);
   \draw[solid](-0.08,0)node[below]{ $2$ }--+(0,1);
   \draw[ghost](0.92,0)--+(0,1)node[above]{ $2$ };
   \draw[solid](0.84,0)node[below]{ $3$ }--+(0,1);
 \end{tikzpicture}
 ]
    '''
    if verbose:
        def vprint(*args): print(args)
    else:
        def vprint(*args): pass

    cart = CartanType(cart)
    R = RootSystem(cart)
    if not isinstance(L, list):
        L = [L]
    wt = sum(R.weight_space().basis()[k] for k in L)
    crystal = crystals.LSPaths(wt)
    G = crystal.digraph(depth=n)
    vprint('Asking sage to construct the crystal')

    # highest weight vector
    vlam = crystal.highest_weight_vector()
    idempotents = [] # will hold the idempotents indexed by the paths of length n
    length = 0       # length of the path
    paths = [ {'vertex': vlam, 'path':[]} ]
    found_new_path = True
    while paths != []:
        p = paths.pop()
        vprint(f'Looking for paths from {p}')
        for _,w,i in G.edges(vertices=[ p['vertex'] ]):
            vprint(f' - there is an {i}-edge to {w}')
            new_path = {'vertex': w, 'path': p['path']+[i]}
            if len(new_path['path']) == n:
                id = KLRWIdempotentDiagram(cart, L, path=new_path['path'], vertex=new_path['vertex'])
                idempotents.append( id )
            else:
                paths.append( new_path )

    idempotents.sort(key = lambda d: str(d._vertex))

    if latex:
        global __KLRW_DIAGRAM_TIKZ__
        output = __latex_file__.format(
            preamble = __KLRW_DIAGRAM_TIKZ__,
            body     = '\n'.join(d._latex_(labelled=True, preamble=False) for d in idempotents)
        )
        if isinstance(latex, str):
            with open(latex,'w') as file:
                file.write( output )
        else:
            return output

    else:
        return idempotents

def tikz_perm(w, top=None, bottom=None):
    r'''
    Return TikZ code for the permutation `w`.

    EXAMPLES:

        sage: print( '\n'.join(tikz_perm(w) for w in SymmetricGroup(6).some_elements()) )
        \begin{tikzpicture}\foreach \d in {1, 2, 3, 4, 5, 6}{\node[circle,fill=black,minimum width=3pt, inner sep=0pt](t\d)at(\d,1){};\node[circle,fill=black,minimum width=3pt, inner sep=0pt](b\d)at(\d,0){};}\draw (t1)--(b2)  (t2)--(b1)  (t3)--(b3)  (t4)--(b4)  (t5)--(b5)  (t6)--(b6);\end{tikzpicture}
        \begin{tikzpicture}\foreach \d in {1, 2, 3, 4, 5, 6}{\node[circle,fill=black,minimum width=3pt, inner sep=0pt](t\d)at(\d,1){};\node[circle,fill=black,minimum width=3pt, inner sep=0pt](b\d)at(\d,0){};}\draw (t1)--(b1)  (t2)--(b3)  (t3)--(b2)  (t4)--(b4)  (t5)--(b5)  (t6)--(b6);\end{tikzpicture}
        \begin{tikzpicture}\foreach \d in {1, 2, 3, 4, 5, 6}{\node[circle,fill=black,minimum width=3pt, inner sep=0pt](t\d)at(\d,1){};\node[circle,fill=black,minimum width=3pt, inner sep=0pt](b\d)at(\d,0){};}\draw (t1)--(b1)  (t2)--(b2)  (t3)--(b4)  (t4)--(b3)  (t5)--(b5)  (t6)--(b6);\end{tikzpicture}
        \begin{tikzpicture}\foreach \d in {1, 2, 3, 4, 5, 6}{\node[circle,fill=black,minimum width=3pt, inner sep=0pt](t\d)at(\d,1){};\node[circle,fill=black,minimum width=3pt, inner sep=0pt](b\d)at(\d,0){};}\draw (t1)--(b1)  (t2)--(b2)  (t3)--(b3)  (t4)--(b5)  (t5)--(b4)  (t6)--(b6);\end{tikzpicture}
        \begin{tikzpicture}\foreach \d in {1, 2, 3, 4, 5, 6}{\node[circle,fill=black,minimum width=3pt, inner sep=0pt](t\d)at(\d,1){};\node[circle,fill=black,minimum width=3pt, inner sep=0pt](b\d)at(\d,0){};}\draw (t1)--(b1)  (t2)--(b2)  (t3)--(b3)  (t4)--(b4)  (t5)--(b6)  (t6)--(b5);\end{tikzpicture}
        \begin{tikzpicture}\foreach \d in {1, 2, 3, 4, 5, 6}{\node[circle,fill=black,minimum width=3pt, inner sep=0pt](t\d)at(\d,1){};\node[circle,fill=black,minimum width=3pt, inner sep=0pt](b\d)at(\d,0){};}\draw (t1)--(b1)  (t2)--(b2)  (t3)--(b3)  (t4)--(b4)  (t5)--(b5)  (t6)--(b6);\end{tikzpicture}
        \begin{tikzpicture}\foreach \d in {1, 2, 3, 4, 5, 6}{\node[circle,fill=black,minimum width=3pt, inner sep=0pt](t\d)at(\d,1){};\node[circle,fill=black,minimum width=3pt, inner sep=0pt](b\d)at(\d,0){};}\draw (t1)--(b1)  (t2)--(b3)  (t3)--(b4)  (t4)--(b5)  (t5)--(b6)  (t6)--(b2);\end{tikzpicture}
        sage: print( '\n'.join(tikz_perm(w,list("123456"), list("123456")) for w in SymmetricGroup(6).some_elements()) )
        \begin{tikzpicture}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](t\c)at(\c,1){\res};}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](b\c)at(\c,0){\res};}\draw (t1)--(b2)  (t2)--(b1)  (t3)--(b3)  (t4)--(b4)  (t5)--(b5)  (t6)--(b6);\end{tikzpicture}
        \begin{tikzpicture}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](t\c)at(\c,1){\res};}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](b\c)at(\c,0){\res};}\draw (t1)--(b1)  (t2)--(b3)  (t3)--(b2)  (t4)--(b4)  (t5)--(b5)  (t6)--(b6);\end{tikzpicture}
        \begin{tikzpicture}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](t\c)at(\c,1){\res};}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](b\c)at(\c,0){\res};}\draw (t1)--(b1)  (t2)--(b2)  (t3)--(b4)  (t4)--(b3)  (t5)--(b5)  (t6)--(b6);\end{tikzpicture}
        \begin{tikzpicture}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](t\c)at(\c,1){\res};}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](b\c)at(\c,0){\res};}\draw (t1)--(b1)  (t2)--(b2)  (t3)--(b3)  (t4)--(b5)  (t5)--(b4)  (t6)--(b6);\end{tikzpicture}
        \begin{tikzpicture}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](t\c)at(\c,1){\res};}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](b\c)at(\c,0){\res};}\draw (t1)--(b1)  (t2)--(b2)  (t3)--(b3)  (t4)--(b4)  (t5)--(b6)  (t6)--(b5);\end{tikzpicture}
        \begin{tikzpicture}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](t\c)at(\c,1){\res};}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](b\c)at(\c,0){\res};}\draw (t1)--(b1)  (t2)--(b2)  (t3)--(b3)  (t4)--(b4)  (t5)--(b5)  (t6)--(b6);\end{tikzpicture}
        \begin{tikzpicture}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](t\c)at(\c,1){\res};}\foreach \res [count=\c from 0] in {1,2,3,4,5,6}{\node[circle,draw,inner sep=1pt](b\c)at(\c,0){\res};}\draw (t1)--(b1)  (t2)--(b3)  (t3)--(b4)  (t4)--(b5)  (t5)--(b6)  (t6)--(b2);\end{tikzpicture}
    '''
    W = w.parent()
    if top is None or bottom is None:
        dots=r'\foreach \d in %s{\node[circle,fill=black,minimum width=3pt, inner sep=0pt](t\d)at(\d,1){};\node[circle,fill=black,minimum width=3pt, inner sep=0pt](b\d)at(\d,0){};}' % W.domain()
        strings = ' '.join(r' (t%s)--(b%s)' % (d,w(d)) for d in W.domain())
        return r'\begin{tikzpicture}%s\draw%s;\end{tikzpicture}' % (dots, strings)

    top = r'\foreach \res [count=\c from 0] in {%s}{\node[circle,draw,inner sep=1pt](t\c)at(\c,1){\res};}' % ','.join(top)
    bottom = r'\foreach \res [count=\c from 0] in {%s}{\node[circle,draw,inner sep=1pt](b\c)at(\c,0){\res};}' % ','.join(bottom)
    strings = ' '.join(r' (t%s)--(b%s)' % (d,w(d)) for d in W.domain())
    return r'\begin{tikzpicture}%s\draw%s;\end{tikzpicture}' % (top+bottom, strings)
