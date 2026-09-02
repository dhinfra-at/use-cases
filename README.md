# DHInfra cluster use cases

Research pipelines that run on the DHInfra cluster: notebooks, scripts, and small sample
datasets you can open and run.

Start with the console documentation at
[`console.dhinfra.uni-graz.at/console/docs`](https://console.dhinfra.uni-graz.at/console/docs).
It already carries a lot of example code — getting a session, requesting a GPU, loading
data, running a job. This repository holds the larger examples that build on it.

This repository covers the integrated cluster environment: JupyterHub, GPUs, batch jobs,
shared storage, the model gateway. The other services on [dhinfra.at](https://dhinfra.at) —
digitisation, scanning hardware, cameras — are out of scope. A use case belongs here if
someone can reproduce it by logging in to the cluster.

## Candidates and curated

`candidates/` holds submissions. Anyone may add one by pull request, and anyone may open a
pull request against an existing one. A candidate is unreviewed: it may not run as
written, and its licence statement may be incomplete.

`curated/` holds what the DHInfra Ops team in Graz has reviewed. Review means the use case
runs on the cluster, its README is enough to follow, its licence statement is correct, and
its files hold nothing that should not be public.

Each use case lives in `NNN-slug/`. The number is its id, it does not change, and a curated
copy keeps it, so `candidates/004-…` and `curated/004-…` are the same use case. The two
copies may be identical; the folder records the review, not a change to the content. Pick
the next free number when you submit. If two pull requests pick the same one, the Ops team
renumbers on merge.

Curation happens in batches, alongside everything else the Ops team does. A candidate that
has been here a while is waiting, not rejected. Only the Ops team writes to `curated/`. A
pull request against an existing candidate is welcome and is the fastest way to get it
curated — fix a broken path, add the missing licence file, clear the notebook outputs.

## How to submit

1. Fork the repository and branch off `main`.
2. Create `candidates/NNN-your-slug/`.
3. Put your code, a `README.md`, and enough sample data to run it in that folder. Copy
   [`USE-CASE-TEMPLATE.md`](USE-CASE-TEMPLATE.md) as your `README.md` and fill it in.
4. Work through the checklist below.
5. Open the pull request against `main`.

Keep the sample data small. It is there to make the example runnable. Point at the full
source — an IIIF endpoint, a Zenodo record, a repository — and ship a slice of it.

## Licence and permission

By opening a pull request you place your submission under these terms and confirm you are
entitled to do so.

- **Code** — [Apache-2.0](LICENSE), the repository default. A folder may ship its own
  permissive licence file instead, and that file then governs it.
- **Material** — data, images, models, documentation: under a Creative Commons licence you
  name explicitly, in your README and in a `LICENSE.txt` beside the data. `CC BY 4.0` and
  `CC BY-SA 4.0` are the usual choices. "Creative Commons" on its own is not a licence.
- **Third-party material** — name its source, its licence, and the attribution it
  requires. Do not re-license someone else's data.
- **Duration** — the permission is indefinite and irrevocable. DHInfra and the people who
  use the cluster may keep, run, adapt, and redistribute the material for as long as the
  service exists, under the licence you named. Withdrawing a use case later removes it from
  this repository; it does not reach the copies already made.

If you cannot grant that for part of your work, leave that part out and describe it in the
README instead.

## Checklist before you push

Once it is in a public repository it is public, and deleting the file later leaves it in
the git history.

**Never commit** `.env` files or any file holding credentials; passwords, API keys, access
tokens, session cookies, `Authorization` headers; SSH or TLS private keys, service-account
JSON, keytabs; connection strings with a password in them; internal hostnames, IP
addresses, or non-public ports.

Read credentials from the environment instead. `candidates/002-anno-download` shows the
pattern: `IIIF_USER` and `IIIF_PASS` come from a `.env` outside the repository, and the
script exits with a clear message when they are missing.

**Also check:**

- **Notebook outputs.** Clear them, or read every one first. Cell output keeps whatever was
  on screen: file paths, sample records, tracebacks.
  `jupyter nbconvert --clear-output --inplace your.ipynb`
- **Absolute paths.** `/home/yourname/…` and `C:\Users\yourname\…` publish an account name
  and a directory layout. Use relative paths, or a variable at the top of the file with a
  comment saying what to set it to.
- **The data.** Personal data, unpublished sources, embargoed material, and anything a
  third party gave you under conditions do not belong here. If you are unsure you may
  republish it, you may not.
- **Names.** Colleagues, students, and interview subjects turn up in file names, metadata,
  and CSV columns more often than people expect.

If something did get pushed, tell the Ops team and treat the credential as compromised.
Rotate it first; cleaning the git history comes after.

## Staging on the cluster

Once the process is running, the Ops team will stage the curated use cases on the cluster,
so that a use case appears in your home drive without you fetching it. This is not live
yet, and this section will be updated when it is. It changes nothing about how you may use
the material: clone the repository, copy a notebook, take one function.

## Contact

Open an issue here, or reach the DHInfra Ops team in Graz. For the cluster itself —
sessions, GPUs, storage, quotas — the
[console documentation](https://console.dhinfra.uni-graz.at/console/docs) answers faster.
