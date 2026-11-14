# The Knots You Cannot Read

## Classifier check

The safetensors file contains `embedding.weight` with shape `[256, 32]`, `classifier.weight` with shape `[3, 32]`, and `classifier.bias` with shape `[3]`. The classifier mean-pools the 32-dimensional embeddings for the input's raw ASCII bytes, then applies the linear classifier. On all 2,400 supplied records, this reproduces 2,290 correct predictions (95.42%).

The corpus uses only byte values 32 and 97 through 121; its maximum input byte is 121. The classifier therefore never looks up embedding row 128 when processing this corpus.

## Payload location and encoding

The flag is stored in `embedding.weight[128, 0:18]`, an embedding row outside the corpus input alphabet. The 18 float32 values encode the flag's ASCII bytes using:

```text
weight = (ASCII_code - 128) / 1024
ASCII_code = round(weight * 1024 + 128)
```

The values decode exactly to `SIGMOID_PROVENANCE`. Row 128 is unreachable from the supplied corpus, so these values do not affect the classifier's predictions on those records.

## Extraction

Run `python extract_flag.py` from this folder. The script reads the safetensors header, extracts the embedding tensor, decodes the values in row 128, and checks the required `SIGMOID_<UPPERCASE_TERM>` format before printing the flag.
