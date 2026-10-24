# Upload this repository

The ZIP extracts to one `hybrid-multi-stego/` folder. The release is a prepared
local upload tree; it has no remote credentials or generated Git history.

To create a new remote repository, choose an empty repository on GitHub, then
run these commands inside the extracted folder, replacing `YOUR-REPOSITORY-URL`:

```sh
python tools/verify_release.py
git init
git add .
git commit -m "Add manuscript experiment code and checked evidence"
git branch -M main
git remote add origin YOUR-REPOSITORY-URL
git push -u origin main
```

To update an existing repository, first clone it and create a branch. Copy this
tree into that clone, review the diff and preserve the existing `.git/`, licence
notices, remote and unrelated work. Do not replace a remote branch by force.
The recorded original project URL is
<https://github.com/Omego-Obinna/hybrid-multi-stego>.

Keep large images, checkpoints, private fixture keys, virtual environments and
runtime output outside ordinary Git. `.gitignore` protects the standard local
artifact directories; it is still necessary to review newly staged files.
After intentional documentation changes, run `python tools/update_checksums.py`
before the integrity check and commit. This refreshes the release tree hashes;
it does not rewrite the original copied-source provenance. Changes to a frozen
source or copied record require a separately documented version, not merely a
new checksum.

The included score arrays and compact numerical records belong in ordinary Git.
See `DATA_AND_ARTIFACTS.md` for the limits of the code/evidence deposit.
