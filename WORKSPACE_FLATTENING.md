# Workspace Flattening

This workspace currently contains two separate Git repositories:

- Outer workspace: `E:\llm-free-memory-fusion-forgetting-master`
- Inner experiment repo: `E:\llm-free-memory-fusion-forgetting-master\llm-free-memory-fusion-forgetting-master`

The inner repository is the active experiment codebase. The outer workspace mainly carries paper-writing and reference materials.

## Recommended Non-Destructive Flattening

Use the inner repository as the base, then copy the outer-only materials into a new flat workspace:

```powershell
pwsh -ExecutionPolicy Bypass -File E:\llm-free-memory-fusion-forgetting-master\prepare_flat_workspace.ps1
```

Default output:

```text
E:\llm-free-memory-fusion-forgetting-unified
```

## What Gets Copied

Base workspace copied from the inner repo:

- `exp`
- `idea`
- `.git`
- `.gitignore`
- `README.md`

Additional materials copied from the outer workspace:

- `.vexp`
- `.vscode`
- `analytical_plots`
- `method_diagrams`
- `reference paper`
- `writing`
- `PAPER_BLUEPRINT.md`
- `upload.py`

## What Stays Untouched

The script does not delete or modify either original workspace.

It intentionally skips:

- the nested directory itself
- temporary probe files such as `tmp_test_write*.txt`

## After Verification

Once the new flat workspace is confirmed usable, you can switch daily work to:

```text
E:\llm-free-memory-fusion-forgetting-unified
```

Only after that should you consider archiving or deleting the old nested layout.
