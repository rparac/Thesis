% Background Knowledge

value("1, 1", X) :- value(1, 1, X).
value("1, 2", X) :- value(1, 2, X).
value("1, 3", X) :- value(1, 3, X).
value("1, 4", X) :- value(1, 4, X).

value("2, 1", X) :- value(2, 1, X).
value("2, 2", X) :- value(2, 2, X).
value("2, 3", X) :- value(2, 3, X).
value("2, 4", X) :- value(2, 4, X).

value("3, 1", X) :- value(3, 1, X).
value("3, 2", X) :- value(3, 2, X).
value("3, 3", X) :- value(3, 3, X).
value("3, 4", X) :- value(3, 4, X).

value("4, 1", X) :- value(4, 1, X).
value("4, 2", X) :- value(4, 2, X).
value("4, 3", X) :- value(4, 3, X).
value("4, 4", X) :- value(4, 4, X).

% Columns
col("1, 1", 1).
col("1, 2", 2).
col("1, 3", 3).
col("1, 4", 4).

col("2, 1", 1).
col("2, 2", 2).
col("2, 3", 3).
col("2, 4", 4).

col("3, 1", 1).
col("3, 2", 2).
col("3, 3", 3).
col("3, 4", 4).

col("4, 1", 1).
col("4, 2", 2).
col("4, 3", 3).
col("4, 4", 4).

% Rows
row("1, 1", 1).
row("1, 2", 1).
row("1, 3", 1).
row("1, 4", 1).

row("2, 1", 2).
row("2, 2", 2).
row("2, 3", 2).
row("2, 4", 2).

row("3, 1", 3).
row("3, 2", 3).
row("3, 3", 3).
row("3, 4", 3).

row("4, 1", 4).
row("4, 2", 4).
row("4, 3", 4).
row("4, 4", 4).


% Blocks
block("1, 1", 1).
block("1, 2", 1).
block("2, 1", 1).
block("2, 2", 1).

block("1, 3", 2).
block("1, 4", 2).
block("2, 3", 2).
block("2, 4", 2).

block("3, 1", 3).
block("3, 2", 3).
block("4, 1", 3).
block("4, 2", 3).

block("3, 3", 4).
block("3, 4", 4).
block("4, 3", 4).
block("4, 4", 4).

neq(X, Y) :- cell(X), cell(Y), X != Y.
num(1..4).
row(1..4).
col(1..4).
block(1..4).
cell(C) :- value(C, _).

selected(valid) :- not selected(invalid).
