# Vietnam — sneaker open-set recognition

Classifies sneaker photos as `nike`, `adidas` or `converse`, or `unknown` for any other shoe.

## Run
1. Put `vietnam_data.zip` in Google Drive at `MyDrive/vietnam/`.
2. Open `train_simple_colab.ipynb` in Colab and do **Runtime → Run all**.
   - If `MyDrive/vietnam/output_simple/model.pt` exists, it lowers that model's `tau` by 0.03 (seconds, no GPU)
     and writes `output_simple/SUBMIT_THIS/model.pt`. This is the version that passed (tau 0.92, saved locally as `model.pt`).
   - Otherwise it trains a new model (~15 min, T4 GPU) and saves it to `output_simple/run_<date>_<time>/`.
3. Submit `model.pt`.

## How it works
Pretrained ConvNeXt-Tiny with 3 outputs. The evaluator predicts the top brand, or `unknown` when the top probability is
below `tau` (stored in `model.pt`). Unused blocks are stored as zeros so the fp16 file is about 8.5 MB (limit 10 MB).
Details are in the notebook's first cell.
