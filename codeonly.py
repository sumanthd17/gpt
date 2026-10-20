"""Print a Python file with all comments and docstrings removed, formatting kept.

Usage:
    python codeonly.py train_gpt2.py                         # print to terminal
    python codeonly.py train_gpt2.py > train_gpt2.code.py    # code-only copy to open in your editor

The original file is never modified.
"""
import io
import sys
import tokenize

src = open(sys.argv[1]).read()
drop = set()  # (row, col) of tokens to blank out
prev = tokenize.NEWLINE
for tok in tokenize.generate_tokens(io.StringIO(src).readline):
    if tok.type == tokenize.COMMENT:
        drop.add(tok.start)
    # a string that is a whole statement on its own = docstring
    elif tok.type == tokenize.STRING and prev in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
        drop.add(tok.start)
    if tok.type not in (tokenize.NL, tokenize.COMMENT):
        prev = tok.type

lines = src.splitlines(keepends=True)
out = list(lines)
for tok in tokenize.generate_tokens(io.StringIO(src).readline):
    if tok.start in drop:
        (r1, c1), (r2, c2) = tok.start, tok.end
        for r in range(r1, r2 + 1):
            out[r - 1] = None if r != r1 or not lines[r - 1][:c1].strip() else lines[r - 1][:c1].rstrip() + "\n"

blank = 0
for line in out:
    if line is None:
        continue
    if not line.strip():
        blank += 1
        if blank > 2:
            continue
    else:
        blank = 0
    sys.stdout.write(line)
