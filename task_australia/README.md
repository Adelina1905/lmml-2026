# task_australia: The Drunken Mario Adventures

Flag: `SIGMOID_TANH` (see `FLAG.txt`)

The trained Marios' odd habit is which of the 20 coins they collect. The fitness
function in the compiled `lab.fitness` module gives every coin a sign:
`format(919247 ^ spread(COIN_SITES), '020b')` with
`spread: h = (h*131 + x*7 + i) & 0xFFFFF`. A `1` means +25 (the agent jumps up for the coin)
and a `0` means -25 (it walks under the coin).
The 20 take/skip bits `10100 00001 01110 01000` split into four 5-bit letters (A=1):
**T A N H**.

## Run
```
pip install -r requirements.txt
python solve.py path/to/marathon-lab-windows-x86_64.exe
```
The exe (21 MB) is not committed; download it from the task page.
`solve.py` unpacks the Nuitka onefile payload, parses the constants blob of the compiled
game (`main.dll`), rebuilds the per-coin signs, decodes the word and writes `FLAG.txt`
and `results/report.txt`.

`results/` also holds screenshots of the live training run: taken coins disappear and
skipped coins stay, matching the bits.
