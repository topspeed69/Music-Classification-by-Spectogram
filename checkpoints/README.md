# Checkpoints

✅ This folder is intentionally kept empty in the repository (and may be committed with a placeholder) to allow storing model checkpoint files during training or when saving final model weights.

Why keep it empty?
- Checkpoint files (commonly large binary files like `.pth`, `.pt`, `.ckpt`) are usually large and not suitable for version control in Git. Instead, we keep this directory present so users and CI can save/load files into a well-known location.

What to put here locally
- Trained model weights: `best_model.pth`, `epoch_10.pth`, etc.
- Checkpoint metadata or training state files if needed.

Best practices
- Keep checkpoint files out of Git history. Use a remote object store (S3, GCS), an artifacts server, or a dedicated release (GitHub Releases) to share large files.
- If you need to make sure the folder exists in a clean clone, add a small placeholder file (for example `.gitkeep`) or commit this `README.md` which documents the reasoning.
- Use `.gitignore` to ensure large files don't accidentally get committed. Example entry:

  ```text
  /checkpoints/*.pth
  /checkpoints/*.pt
  /checkpoints/*.ckpt
  ```

Using checkpoints in code
- Typical pattern when training or evaluating: set CHECKPOINT_DIR to this folder and save/load from it. Example in Python:

  ```py
  # training
  torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pth'))

  # loading
  model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth')))
  ```

Security & sharing
- Avoid committing checkpoints that contain private data, secrets, or models you don't have rights to share.
- Use releases or an approved artifact storage for sharing trained models across collaborators.

If you want this folder to remain empty in the remote repository while still being present in local clones, keep an empty `.gitkeep` file and add `README.md` (this file) so the intent is clear.

Questions or requests
- Need a helper script to upload/download checkpoints to a cloud store? Open an issue or create a PR and we can add tooling.

---
Last updated: 2025-11-29
