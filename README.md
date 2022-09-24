GradedDimKLR
============

[SageMath](https://www.sagemath.org/SageMath) code for computing the dimensions of the weight spaces of the cyclotomic KLR algebras attached to symmetrisable quivers using the amazing formula

```math
    \dim_q e(i)R^\Lambda_n e(j)
          = \sum_{w\in S_{i,j}}
              \prod_{t=1}^n[N^\Lambda(w,i,t)]_{i_t} q_{i_t}^{N^\Lambda(i,t)-1}
```

from the paper [Graded dimensions and monomial bases for the cyclotomic quiver Hecke algebras](https://arxiv.org/abs/2108.05508G) of Hu and Shi. Here, $e(i)R^\Lambda_ne(j)$ is the weight space of the KLR algebra $R^\Lambda_n$ determined by the sequences $i,j\in I^n$ and $\Lambda$ is a dominant weight for the corresponding quiver, which has vertex set $I$. See the Hu-Shi paper for the unexplained notation in the formula.

The attached code provides a single command ``klr_cyclotomic_dimension`` that uses the Hu-Shi formula to compute the dimension. The syntax of this command is:

```python
   klr_cyclotomic_dimension(C, L, i, [j, verbose])
```

where:

* ``C`` specifies the *Cartan type* of the quiver. This can either be a list
  of the form ``['A', 3]`` (finite type $A_3$), ``['A', 3, 1]`` (affine type
  $A_3^{(1)}$), ``['B',4]`` (finite type $B_4$), etc or ``C`` can be a [SageMath](https://www.sagemath.org/SageMath) Cartan type, such as ``CartanType(['D',3,1])``.

* ``L`` is a *list* that specifies the dominant weight. For example,
  ``[0,2,2,3]`` represents the dominant weight
  $\Lambda_0+2\Lambda_2+\Lambda_3$.

* ``i`` is an `n`-tuple of vertices of the quiver

* ``j`` is an `n`-tuple of vertices of the quiver. If ``j`` is omitted then
  ``j`` is set equal to ``i``.

* ``verbose`` an optional parameter that when set to ``True`` causes additional
  information from the Hu-Shi formula to be printed.

Examples
--------

```python
    sage: klr_cyclotomic_dimension(['D',4],[2], [2,3,4,1], verbose=True)
    Subgroup of permutations = < () >
    N(1,t)-1: 0  0  0  0
    N(w,t):   1  1  1  1
    X(w): 1
    1
    sage: klr_cyclotomic_dimension(['D',4],[2], [2,3,4,1], [2,3,4,1], verbose=True)
    Subgroup of permutations = < () >
    N(1,t)-1: 0  0  0  0
    N(w,t):   1  1  1  1
    X(w): 1
    1
    sage: klr_cyclotomic_dimension(['D',4],[2], [2,3,4,1], [2,4,3,1], verbose=True)
    Subgroup of permutations = < (1,2) >
    N(1,t)-1: 0  0  0  0
    N(w,t):   1  1  1  1
    X(w): 1
    1
    sage: klr_cyclotomic_dimension(['A',2,1],[0], [0,1,2], verbose=True)
    Subgroup of permutations = < () >
    N(1,t)-1: 0  0  1
    N(w,t):   1  1  2
    X(w): q^2 + 1
    q^2 + 1
    sage: klr_cyclotomic_dimension(['A',2,1],[0], [0,2,1], verbose=True)
    Subgroup of permutations = < () >
    N(1,t)-1: 0  0  1
    N(w,t):   1  1  2
    X(w): q^2 + 1
    q^2 + 1
    sage:
    sage: klr_cyclotomic_dimension(['A',2,1],[0], [0,1,2], [0,2,1], verbose=True)
    Subgroup of permutations = < (1,2) >
    N(1,t)-1: 0  0  1
    N(w,t):   1  1  1
    X(w): q
    q
    sage: klr_cyclotomic_dimension(['A',1],[1,1],[1],[1], verbose=True)
    Subgroup of permutations = < () >
    N(1,t)-1: 1
    N(w,t):   2
    X(w): q^2 + 1
    q^2 + 1
    sage: klr_cyclotomic_dimension(['A',1],[1,1],[1],[1], verbose=True)
    Subgroup of permutations = < () >
    N(1,t)-1: 1
    N(w,t):   2
    X(w): q^2 + 1
    q^2 + 1
    sage: klr_cyclotomic_dimension(['B',3],[2], [2,3,3,2,1], verbose=True)
    Subgroup of permutations = < (), (1,2), (0,3), (0,3)(1,2) >
    N(1,t)-1: 0  1 -1  0  1
    N(w,t):   1  2  0  1  2
    X(w): 0
    N(w,t):   1  2  2  1  2
    X(w): (q^4 + 1)*(q^2 + 1)^2/q^2
    N(w,t):   1  0 -2  1  2
    X(w): 0
    N(w,t):   1  0  0  1  2
    X(w): 0
    (q^4 + 1)*(q^2 + 1)^2/q^2
```

Usage
-----

* With a local installation of [SageMath](https://www.sagemath.org/), start [SageMath](https://www.sagemath.org/) and attach the file ``graded_dim_klr`` using:

```python
   sage: %attach graded_dim_klr
   sage: klr_cyclotomic_dimension(['A',3],[2], [2,3,3,2,1], [2,3,2,3,1])
   sage: klr_cyclotomic_dimension(['B',3],[2], [2,3,3,2,1], verbose=True)
```

* Using [SageMathCell](https://sagecell.sagemath.org/), cut-and-paste the code from ``graded_dim_klr`` into a cell
  and then type the ``klr_cyclotomic_dimension`` commands into the bottom of the
  cell.


```python
   klr_cyclotomic_dimension(['A',3],[2], [2,3,3,2,1], [2,3,2,3,1])
   klr_cyclotomic_dimension(['B',3],[2], [2,3,3,2,1], verbose=True)
```

Author
------
Andrew Mathas
Copyright (C) 2022

------------

GNU General Public License, Version 3, 29 June 2007

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License [GPL](http://www.gnu.org/licenses/gpl.html
) as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
