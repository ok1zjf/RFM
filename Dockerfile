# Runtime image for rfm/: init.sh -> train_test.sh / test.sh, run as one-off
# `docker run --rm` invocations against a single host directory bind-mounted
# at /app/data (see README.md's "Installation" > "Docker"). weights/,
# datasets/, and outputs/ are symlinks into /app/data/{weights,datasets,
# outputs} so that one mount covers all three.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# curl/unzip: download_weights.sh + download_datasets.sh. file: download_weights.sh's
# HTML-response detection (falls back to a `head`-based check if missing, but
# keep it for reliability). ca-certificates: HTTPS downloads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        unzip \
        file \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pillow from Anaconda's pkgs/main, built against IJG libjpeg 9e - the exact
# build (hcea889d_0, jpeg-9e h5eee18b_3) that produced the paper's results.
# The base image's pip Pillow (like every PyPI/conda-forge build) bundles
# libjpeg-turbo instead, which decodes JPEGs to slightly different pixels and
# so shifts every result (not bit-exact). Installed before requirements.txt
# so its pillow==11.1.0 pin is already satisfied and pip leaves it alone.
RUN pip uninstall -y pillow \
    && conda install -y --override-channels -c https://repo.anaconda.com/pkgs/main \
        "pillow=11.1.0=py311hcea889d_0" "jpeg=9e" \
    && conda clean -afy \
    && python -c "import PIL.Image as I, PIL.features as F; \
assert I.core.jpeglib_version.startswith('9') and not F.check_feature('libjpeg_turbo'), 'wrong libjpeg'"

# torch/torchvision already come from the base image at the pinned version;
# only install what's left. COPY just this file first so the pip layer stays
# cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x *.sh

# The dataset converters (convert_<dataset>.py) alone use a second Pillow, the
# PyPI build with libjpeg-turbo, from /app/.prep_pillow - the library that
# produced the published datasets/ images (see turbo_pillow.py). Baked in
# here because /app isn't writable for the `--user` the container runs as.
RUN python turbo_pillow.py

# Stash the dataset split/label CSVs somewhere outside datasets/ itself, so
# they survive a user bind-mounting an empty host directory over
# datasets/ (which would otherwise shadow the copies baked in above) -
# prepare_datasets.sh restores them from here if it finds them missing.
RUN mkdir -p /opt/rfm-dataset-csvs && cp datasets/*.csv /opt/rfm-dataset-csvs/

# Collapse weights/, datasets/, outputs/ into symlinks onto one shared
# directory (/app/data), so a single `-v host_dir:/app/data` bind mount
# populates all three instead of needing three separate -v flags. The real
# datasets/ (its CSVs already preserved in the stash above) is removed
# first; weights/ and outputs/ were never baked in to begin with
# (.dockerignore excludes both). The /app/data/* subdirs are pre-created so
# this also works unmodified if nothing is mounted there at all - but if
# something IS mounted, it must already contain weights/datasets/outputs
# subdirs (README.md's `mkdir -p RFM/{weights,datasets,outputs}` setup
# step), since mounting replaces /app/data entirely: without them, the
# symlinks dangle and even `mkdir -p` refuses to create through them.
RUN rm -rf datasets \
    && mkdir -p /app/data/weights /app/data/datasets /app/data/outputs \
    && ln -s /app/data/weights weights \
    && ln -s /app/data/datasets datasets \
    && ln -s /app/data/outputs outputs

# No ENTRYPOINT/CMD beyond the base image's default shell - there is no
# persistent container; every script runs as its own `docker run --rm`
# (see README.md's "Installation" > "Docker").
