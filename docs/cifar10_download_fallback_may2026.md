# CIFAR-10 download fallbacks (May 2026)

## Why this exists

`torchvision.datasets.CIFAR10` downloads the Python version of CIFAR-10 from a single hard-coded URL on the University of Toronto site (`cifar-10-python.tar.gz`). That host occasionally returns HTTP 503 or is temporarily unavailable, which breaks fresh Colab or CI runs that need `download=True`.

The archive is identified by an MD5 checksum (`c58f30108f718f92721af3b95e74349a`) published with the official dataset page; torchvision uses the same checksum in `torchvision/datasets/cifar.py` so any mirror must be byte-identical to the official file.

## What the project does

`load_cifar10_datasets` in `src/gradient_ascent/data.py` tries several URLs in order: the canonical HTTPS link, the same path over plain HTTP (sometimes useful when TLS or a CDN edge fails, as discussed in the torchvision issue tracker), and a third-party directory mirror that lists the same `cifar-10-python.tar.gz` file. Between attempts it keeps the existing exponential backoff for transient errors.

If you host a private copy (e.g. on cloud storage), set a comma-separated list in the environment variable `GRADIENT_ASCENT_CIFAR10_URLS` so your URL is tried first—or as the only source if you provide the full ordered list you want.

## References

- Alex Krizhevsky, “CIFAR-10 and CIFAR-100 datasets” — official page with version table and MD5 for the Python tarball: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- PyTorch Vision `CIFAR10` implementation (default URL, filename, `tgz_md5`): [torchvision.datasets.cifar](https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py)
- PyTorch Vision issue #5039 — users report SSL or availability problems with the default URL and switch to HTTP or alternative mirrors: [github.com/pytorch/vision/issues/5039](https://github.com/pytorch/vision/issues/5039)
- Optional mirror directory listing `cifar-10-python.tar.gz`: [data.brainchip.com dataset-mirror/cifar10](https://data.brainchip.com/dataset-mirror/cifar10/) (third-party; same filename as torchvision expects)
