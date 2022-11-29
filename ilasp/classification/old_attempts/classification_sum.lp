% How much weight does some concept contribute to a label
label(ball).
label(strike).
label(hit).

score(1..10).

concept(1).
concept(2).

contributes(ball, 2, 1) :- concept(1).
contributes(hit, 1, 1) :- concept(1).
contributes(strike, 10, 2) :- concept(2).


score(X, Z) :- label(X), Z = #sum{W : contributes(X, W, _)}.
selected(X) :- score(X, Wx), #max{W : score(_, W)} = Wx.

