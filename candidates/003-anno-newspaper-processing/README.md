# ANNO Newspaper Processing Pipeline

This use case turns scanned historical Austrian newspaper pages into searchable, machine-readable text. It takes a list of newspapers from **ANNO** (AustriaN Newspapers Online, the digitised newspaper collection of the Austrian National Library), downloads the page images, works out where the text sits on each page, reads the text, and optionally cleans it up.

Everything is driven from a single Jupyter notebook, `ANNO_pipeline.ipynb`. 

## Table of contents

1. [What the pipeline does](#what-the-pipeline-does)
2. [What you need](#what-you-need)
3. [Files you need to download](#files-you-need-to-download)
4. [Setup](#setup)
5. [Prepare your input list](#prepare-your-input-list)
6. [Starting the notebook](#starting-the-notebook)
7. [Settings to check before you run](#settings-to-check-before-you-run)
8. [Walking through the notebook](#walking-through-the-notebook)
9. [What the pipeline produces](#what-the-pipeline-produces)
10. [Troubleshooting](#troubleshooting)
11. [Credits and tools used](#credits-and-tools-used)

---

## What the pipeline does

The notebook runs in four stages. Each stage feeds the next.

1. **Collect pages.** It reads your input list of newspapers, asks the Austrian National Library's IIIF service for each one, and builds a full list of individual page images.
2. **Layout analysis.** For every page, it works out the structure of the page: where the columns are, which blocks are text, which are images, and in what order a human would read them. This step uses a tool called **eynollah** and produces one PAGE-XML file per page.
3. **OCR (text recognition).** It reads the actual words off each page image using **Tesseract** with a model trained for Fraktur. The recognised text is written back into the XML files.
4. **Post-correction (optional).** It pulls the text out of the XML files into a table, tidies up some old typographic characters, and can run the text through a correction model to fix common OCR mistakes. The final result is saved as a spreadsheet file.

You can stop after any stage. The OCR text alone is already usable; the post-correction stage is an extra polish.

---

## What you need

**Hardware**

* One or more **NVIDIA GPUs** with CUDA support. The layout step uses roughly 13 GB of GPU memory and 20 to 40 GB of system RAM per running job. The notebook is set up to run several jobs in parallel, so more GPUs and more RAM mean faster processing.
* Disk space for the **XML output**, not for images. The pipeline does not keep the page images. It downloads them in small batches, processes them, and deletes them straight away, so images only ever take up a small, temporary amount of space. What accumulates on disk is the XML (and the final spreadsheet), which is far smaller than the images would be.
* Some spare **RAM** for the OCR step. That step downloads images into a fast in-memory scratch area rather than to disk, and a setting caps how many it holds at once (see [Settings](#settings-to-check-before-you-run)).

**Software**

* **Linux**, ideally **Ubuntu 24.04**. The setup script is written and tested for it. (Ubuntu 22.04 ships an older Tesseract; see the note at the very bottom of `setup.sh` if you are on 22.04.)
* **conda** or **mamba** already installed. If you have neither, install Miniforge first: https://github.com/conda-forge/miniforge
* **sudo rights**, because the setup installs a few system packages (Tesseract and some libraries).

If you are missing the GPU, the pipeline will not work as written. Everything else the setup script handles for you.

---

## Files you need to download

The easiest way to get everything in one place is to clone the repository with `git`.

Open a terminal, move to the location where you want the project to live, and clone it:

```bash
cd ~/projects
git clone --filter=blob:none --sparse https://github.com/dhinfra-at/use-cases.git
cd use-cases
git sparse-checkout set candidates/003-anno-newspaper-processing
cd candidates/003-anno-newspaper-processing
```

Keep every file together in this **single directory**. Do not split them across folders. The directory should contain at least the following:

| File | What it is |
|------|------------|
| `ANNO_pipeline.ipynb` | The notebook you will run. |
| `setup.sh` | The setup script that builds everything. |
| `eynollah_anno.sh` | Helper script the notebook calls to run the layout step. Downloads pages in batches, runs eynollah, and cleans up. |
| `ocr_pipeline_anno.py` | Helper code the notebook imports for the OCR step. Streams images through Tesseract and deletes them as it goes. |
| `correction_multigpu.py` | Helper code the notebook imports for the post-correction step. |

After setup runs, you will also see a new `model_dir/` folder (the layout models) and, as you use the notebook, folders called `anno_source/`, `Layout/`, `OCR/`, and `logs/`.

---

## Setup

You should already be inside the project directory from the previous step. If you opened a new terminal, move back into it first (for example `cd ~/projects/use-cases/candidates/003-anno-newspaper-processing`). Then run these two commands:

```bash
chmod +x setup.sh
./setup.sh
```

The first command makes the script runnable. The second runs it. This takes a while, because it downloads models and builds two environments. Let it finish.

**What the setup script builds for you:**

1. **`eyn_env`**: a dedicated environment that runs eynollah (the layout tool). The notebook calls into this environment behind the scenes. You do not select it as a kernel.
2. **`anno_env`**: the environment the notebook itself runs in. It contains Jupyter and all the OCR and data libraries. It also registers a Jupyter kernel named **"Python (anno_env)"**.
3. **System Tesseract 5** plus the build tools the OCR library needs.
4. **The eynollah layout models**, downloaded into `model_dir/`.
5. **The frak2021 Fraktur model** for Tesseract, installed where Tesseract can find it.

When it finishes, the script prints a short summary telling you where each piece landed. It is worth reading that summary, especially the line showing the path to `eyn_env`, because you will need it in a moment.

---

## Prepare your input list

The file `data_dh-infra.csv` referenced in the notebook is only a **placeholder**. Before you run anything, add your own list of the newspaper editions you want to process. Keep the same file name (`data_dh-infra.csv`) so the notebook finds it, or change the file name in the first stage of the notebook to match.

The file is a normal CSV (a spreadsheet you can edit in Excel, LibreOffice, or any text editor) and must contain these three columns, with exactly these names in the header row:

| Column | What it holds |
|--------|---------------|
| `title` | The full title of the newspaper, for example *Grazer Volksblatt*. |
| `anno_id` | The identifier of the specific newspaper edition. |
| `iiif_manifest` | The IIIF manifest link for that edition. |

One row per edition. A small example:

```csv
title, anno_id, iiif_manifest
Genossenschafts- und Vereins-Zeitung, guv18911215, https://api.onb.ac.at/iiif/presentation/v3/manifest/11F4D747
```

---

## Starting the notebook

Start Jupyter from the same directory using the notebook environment:

```bash
conda run -n anno_env jupyter lab
```

When the notebook opens, set the kernel to **"Python (anno_env)"** using the kernel selector in the top right. If the notebook opens with a different kernel name (you may see an old name like `jobads_env`), switch it to "Python (anno_env)". The notebook will not work correctly under any other kernel.

---

## Settings to check before you run

A few values in the notebook point at the original author's machine and need to be changed to match yours. They are easy to find. Look for these and edit them once, near the top of each relevant section.

**In the Layout Analysis section (the configuration cell):**

* `EYNOLLAH_PREFIX = Path("/home/lab6/miniforge3/envs/eyn_env")`
  Change this to the real location of your `eyn_env` environment. To find it, run `conda env list` in a terminal and copy the path shown next to `eyn_env`.
* `N_GPUS = 2` and `JOBS_PER_GPU = 2`
  Set `N_GPUS` to how many GPUs you actually have. Adjust `JOBS_PER_GPU` based on your GPU memory (remember each job needs about 13 GB). If unsure, start with `JOBS_PER_GPU = 1`.

**In the OCR section, the OCR run:**

* `tessdata_path = "/usr/share/tesseract-ocr/5/tessdata"`
  This matches the default Tesseract 5 location and is usually correct. If the setup summary reported a different tessdata directory, use that one instead.
* `model_name = "frak2021-0.905"`
  Leave this as is unless you renamed the Tesseract model file.
* `max_in_flight`, `ocr_workers`, `download_workers`
  These control how hard the OCR step works and how much memory it uses. They are sized for a large machine. If the OCR step runs short of memory, lower `max_in_flight` first. If the IIIF server starts refusing requests, lower `download_workers`.

**In the Post-correction section (if you use it):**

* `gpu_ids=[0, 1]`
  List the GPU numbers you want to use for correction. With one GPU, use `[0]`.

If you only change one thing, make it the `EYNOLLAH_PREFIX` path. The layout step depends on it.

---

## Walking through the notebook

The notebook is organised top to bottom. Run the cells in order. Below is what each section does in plain terms.

### 1. Get individual pages from IIIF manifest

This section reads your input file `data_dh-infra.csv` (the list you prepared in [Prepare your input list](#prepare-your-input-list)). For each edition it follows the IIIF manifest link. A manifest is a description of a digitised object that lists all of its pages.

The notebook visits each manifest, with automatic retries if the server is busy, and pulls out the direct image link for every single page along with its year. It builds one big table with a row per page and saves it as `data_dh-infra_pages.csv`. This table is the master list that the rest of the pipeline works from.

### 2. Layout analysis (eynollah)

This is the first heavy step and the one that uses your GPUs. The configuration cell sets your environment path and GPU settings (see [Settings to check](#settings-to-check-before-you-run)). The notebook groups the pages by newspaper and year, writes a small list file for each group, then runs eynollah on each group in parallel.

eynollah looks at each page image and figures out the layout: columns, text blocks, images, separators, and reading order. It writes the result as a PAGE-XML file per page into a `Layout/` folder. It does **not** read any text yet; it only maps the structure.

Behind the scenes, each job downloads its pages in batches of 200, runs eynollah on the batch, then deletes those images before moving to the next batch. Only a small number of images sit on disk at any moment, and they are all removed when the job ends.

The notebook keeps a log for every job and shows a tick or a cross as each one finishes. If a job fails, it tells you which log file to look in. The pipeline also checks which jobs are already done, so if you stop and restart, it only processes what is left.

### 3. OCR (text recognition)

This section has two parts.

**Adjust filepath in XML file to IIIF link.** The layout XML files refer to image files by a local name. Before OCR, the notebook rewrites each of these references to point at the online IIIF image link instead, so the OCR step can fetch the images directly. It reports how many files it updated, skipped, or had trouble with.

**OCR.** The notebook scans the `Layout` folder, skips anything already processed, and runs the recognition pipeline. For each page it downloads the image into a fast in-memory scratch area, reads the text line by line within the regions that eynollah marked, writes the recognised text into a new XML file in an `OCR/` folder, then deletes the scratch image. Nothing is read until layout has run, and no images are kept afterwards. The new XML still stores the original IIIF link, so the result stays portable and the image can always be fetched again later. The settings let you control how many images are downloaded and read at once, which is what keeps memory use in check.

### 4. Post-correction

This final section is optional and has two parts.

**Extracting text from XML files.** The notebook reads every OCR XML file and pulls the text out in proper reading order, one row per text region. It builds a table and adds columns for the newspaper, the year, and the page, worked out from the folder structure.

**Post-correction with hmByT5.** The notebook first normalises some old typographic characters (for example turning ligature characters and old abbreviation marks into modern equivalents). It then splits the text into short pieces and runs them through a correction model to fix common OCR errors. The corrected pieces are stitched back together per region. The final table is saved as `data_dh-infra_postcorrected.csv` (there is also a commented-out line to save it as a Parquet file instead).

> **Note:** the correction model currently wired in is marked as *not recommended*, because its performance is inconsistent on text taken from full-page layout segmentation. Treat this stage as experimental. The OCR output from stage 3 is the dependable result. You can skip stage 4 entirely and still have usable text.

---

## What the pipeline produces

| Output | Where | Stage |
|--------|-------|-------|
| `data_dh-infra_pages.csv` | project directory | Page list (stage 1) |
| PAGE-XML layout files | `Layout/<newspaper>/<year>/` | Layout (stage 2) |
| Per-job logs | `logs/` | Layout (stage 2) |
| PAGE-XML files with recognised text | `OCR/<newspaper>/<year>/` | OCR (stage 3) |
| `data_dh-infra_postcorrected.csv` | project directory | Post-correction (stage 4) |

The PAGE-XML files in `OCR/` are the core result. They hold both the layout and the recognised text, they keep the IIIF link to the original image rather than bundling the image itself, and they work with standard tools for historical text. The final CSV is the most convenient format if you just want the text in a table.

---

## Troubleshooting

**A layout job shows a cross (failed).** Open the matching log file in the `logs/` folder. The most common cause is a wrong `EYNOLLAH_PREFIX` path or not enough GPU memory. Lower `JOBS_PER_GPU` and try again. Finished jobs are skipped on a rerun, so you only retry the failures.

**"Module not found" when importing the helper code.** Make sure `ocr_pipeline_anno.py` and `correction_multigpu.py` are in the same directory as the notebook, and that you are running the "Python (anno_env)" kernel.

**Nothing happens in the layout step, or it cannot find the environment.** Run `conda env list`, confirm `eyn_env` exists, and copy its exact path into `EYNOLLAH_PREFIX`.

**The OCR step finds no files.** Check that the "Adjust filepath" cell ran successfully and that `input_root` points at `"Layout"`, not the old hard-coded path.

**Tesseract cannot find the language model.** Confirm `model_name` matches the installed file and that `tessdata_path` matches the directory reported in the setup summary.

**Out of memory.** Reduce `JOBS_PER_GPU` in the layout step. In the OCR step, the in-memory scratch area is the main consumer, so lower `max_in_flight`, and reduce the worker counts in the OCR and post-correction steps if needed.

**You are on Ubuntu 22.04.** The default Tesseract there is an older version. Read the note at the bottom of `setup.sh` for the extra steps needed to install Tesseract 5.

---

## Credits and tools used

* **ANNO (AustriaN Newspapers Online)** and the IIIF service of the **Austrian National Library** provide the digitised newspaper pages.
* **eynollah**, developed at the Berlin State Library (SBB), performs the document layout analysis and produces PAGE-XML.
* **Tesseract 5** with the **frak2021** Fraktur model (by Stefan Weil, University of Mannheim) performs the OCR.
* **hmByT5** provides the experimental OCR post-correction model.

This notebook orchestrates these tools into a single workflow for processing historical Austrian newspapers.
