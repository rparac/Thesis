% Name, priority (lower=better)
label(strike, 0).
label(foul, 1).
label(out, 2).
label(play, 3).
label(ball, 4).

label(L) :- label(L, _).
learnable_label(L) :- exists_lower_priority(L, _).


% Allow at most one label based to be selected
selected(L) :- label(L, P), conds(L), not higher_priority_selection(L, P).
higher_priority_selection(L, P) :- label(L, P), label(L2, P2), P2 < P, selected(L2).

% Select default label if no higher priority selection is made
selected(L) :- label(L, P), not higher_priority_selection(L, P), not exists_lower_priority(L, P).
exists_lower_priority(L, P) :- label(L, P), label(L2, P2), P2 > P.

concept_id(0..77).

% Need to learn:
% conds(strike)
% conds(foul)
% conds(out)
% conds(play)
