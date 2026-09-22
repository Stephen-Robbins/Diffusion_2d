# Historical master snapshot

This directory preserves the source from the former `master` branch at
`fd5dcf2224b744d1767a89b8a9043b8e91cac395`. That branch had a separate Git history from `main`.
The active repository remains on `main`; this snapshot is historical and is
not a replacement for the current code or a new scientific validation.

All historical commits are parents of the consolidation merge and remain
accessible from `main`. Source, notebooks, bibliography/style files, and small
figures are copied here byte-for-byte. Generated LaTeX build outputs remain in
Git history rather than being duplicated. `source-manifest.json` records every
original path, Git blob, byte count, SHA-256, and whether it is copied here.

Recover any original file with:

```bash
git show fd5dcf2224b744d1767a89b8a9043b8e91cac395:path/to/file
```

The former branch was removed after this snapshot and history were pushed.
