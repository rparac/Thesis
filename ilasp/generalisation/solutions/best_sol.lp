in_generalised_sent(V1) :- dep(dobj,V2,V1).
in_generalised_sent(V1) :- dep(nsubjpass,V1,V2).
in_generalised_sent(V1) :- dep(neg,V2,V1).
in_generalised_sent(V1) :- dep(nsubj,V2,V1).
in_generalised_sent(V1) :- dep(nsubjpass,V2,V1).
in_generalised_sent(V1) :- dep(nsubj,V1,V2).
in_generalised_sent(V1) :- dep(oprd,V2,V1).
in_generalised_sent(V1) :- dep(mark,V2,V1).
in_generalised_sent(V1) :- dep(attr,V2,V1).
in_generalised_sent(V1) :- dep(prt,V2,V1).
 :- root(V1); not in_generalised_sent(V1).
in_generalised_sent(V1) :- in_generalised_sent(V2); dep(det,V2,V1).
in_generalised_sent(V1) :- in_generalised_sent(V2); dep(prep,V1,V2).
0 {in_generalised_sent(V1) } 1 :- dep(poss,V2,V1).
in_generalised_sent(V1) :- in_generalised_sent(V2); dep(pobj,V1,V2).
0 {in_generalised_sent(V1) } 1 :- dep(pobj,V2,V1).
in_generalised_sent(V1) :- in_generalised_sent(V2); dep(auxpass,V2,V1).
in_generalised_sent(V1) :- in_generalised_sent(V2); dep(compound,V2,V1).
in_generalised_sent(V1) :- in_generalised_sent(V2); dep(case,V2,V1).
in_generalised_sent(V1) :- in_generalised_sent(V2); dep(aux,V2,V1).
in_generalised_sent(V1) :- in_generalised_sent(V2); dep(advmod,V2,V1).
0 {in_generalised_sent(V1) } 1 :- in_generalised_sent(V2); dep(amod,V2,V1).
in_generalised_sent(V1) :- in_generalised_sent(V2); dep(acomp,V2,V1).
