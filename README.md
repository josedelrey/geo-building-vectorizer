# geo-building-vectorizer

## Environment

The project uses the Conda environment named `cv`.

Create it with:

```powershell
conda env create -f environment.yml
```

Update an existing environment with:

```powershell
conda env update -n cv -f environment.yml --prune
```

Run project scripts through that environment:

```powershell
conda run -n cv python scripts/check_dataloader.py --config configs/data.yaml --split train
```

For pip-based environments, `requirements.txt` mirrors the direct runtime dependencies.
