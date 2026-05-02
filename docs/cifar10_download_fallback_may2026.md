# CIFAR-10 download fallbacks (May 2026)

## Why this exists

`torchvision.datasets.CIFAR10` downloads the Python version of CIFAR-10 from a single hard-coded URL on the University of Toronto site (`cifar-10-python.tar.gz`). That host occasionally returns HTTP 503 or is temporarily unavailable, which breaks fresh Colab or CI runs that need `download=True`.

The archive is identified by an MD5 checksum (`c58f30108f718f92721af3b95e74349a`) published with the official dataset page; torchvision uses the same checksum in `torchvision/datasets/cifar.py` so any mirror must be byte-identical to the official file.

## What the project does

`load_cifar10_datasets` in `src/gradient_ascent/data.py` tries URLs in order (unless overridden by env): canonical Toronto (`cs.toronto.edu`), then the Brainchip dataset mirror, then the Azure ML public examples blob (same tarball as in Microsoft’s Azure ML pipeline samples). Transient failures use exponential backoff per mirror.

If you host a private copy (e.g. on cloud storage), set a comma-separated list in the environment variable `GRADIENT_ASCENT_CIFAR10_URLS` so your URL is tried first—or as the only source if you provide the full ordered list you want.

## References

- Alex Krizhevsky, “CIFAR-10 and CIFAR-100 datasets” — official page with version table and MD5 for the Python tarball: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- PyTorch Vision `CIFAR10` implementation (default URL, filename, `tgz_md5`): [torchvision.datasets.cifar](https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py)
- Azure ML examples pipeline downloads `cifar-10-python.tar.gz` from `azuremlexamples.blob.core.windows.net`: [azureml-examples `cli/jobs/pipelines/cifar-10/pipeline.yml`](https://github.com/Azure/azureml-examples/blob/main/cli/jobs/pipelines/cifar-10/pipeline.yml)
- Mirror directory listing `cifar-10-python.tar.gz`: [data.brainchip.com dataset-mirror/cifar10](https://data.brainchip.com/dataset-mirror/cifar10/) (third-party)
