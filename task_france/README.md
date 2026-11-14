# task_france: Point par point

**Flag:** `SIGMOID_NEURON`

## How to run
```
pip install -r requirements.txt
python solve.py
```
This writes `labels.csv`, `FLAG.txt` and the files in `results/`.

## Method
1. **Colour of a dot.** The colour is what a traveller weighs before booking: `log(price)`, `availability_365` and `log(reviews_per_month)`, each standardised.
2. **Separate the letters from the city.** For every listing we take its 10 nearest neighbours in colour space and compute the median map distance to them.
   - Inside a planted letter, colour changes smoothly along the strokes, so dots with similar colours are also neighbours on the map (about 0.37 km).
   - For real listings, colour and location are mostly unrelated (about 3.4 km).
   - The histogram has two clear peaks (`results/01_score_histogram.png`). A 2-component Gaussian mixture on the log-score picks the planted dots: 1,592 of them.
3. **One cluster per letter.** We build a mutual k-NN graph (k = 8) on the planted dots, using colour plus location (0.5 km on the map counts as 1 standard deviation of colour). Each connected piece of the graph is a cluster. One small piece (50 dots, 0.01 km away) is joined to the nearest letter. The result is **6 clusters**, one per letter.
4. **Look at each letter.** Inside every cluster the dots fill a flat 2-D sheet in the 5-D space of (lat, lon, log price, availability, log reviews/month). A PCA of the standardised 5-D points shows this: 2 large singular values, 3 small ones. The first two components draw the letter. Each letter is drawn with its own rotation and squash; two letters (R, E) are almost flattened into a line on the map itself, so the 5-D view is needed (`results/03_letters.png`). The first N's reviews/month values are tiny and rounded to 0.01, so its 5-D view is noisy; it is drawn from its map shape instead.
5. **The artist's order.** The letters are put in order by their median **price per night** (the main thing a traveller weighs), cheapest first.

| label | letter | dots | median price | median availability | median reviews/month |
|---|---|---|---|---|---|
| 0 | N | 226 | 97 | 26 | 0.31 |
| 1 | E | 211 | 126 | 303 | 0.98 |
| 2 | U | 294 | 183 | 338 | 1.68 |
| 3 | R | 179 | 250 | 153 | 1.10 |
| 4 | O | 311 | 321 | 250 | 0.40 |
| 5 | N | 320 | 498 | 58 | 3.38 |

Read in price order, the letters spell **NEURON**.

## Files
- `solve.py`: the whole pipeline
- `labels.csv`: `listing_id, label, letter`. The label is the letter's position in the word (price order); -1 means a real listing.
- `FLAG.txt`: the flag
- `results/`: score histogram, overview map, one plot per letter, all letters together, colour-space plots, `report.txt`
- `data/paris_listings.csv`: the provided data
