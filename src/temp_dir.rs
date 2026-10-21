//! Directory where backends write temporary files, such as the CoreML model packages that cannot
//! be loaded from memory. Defaults to the system temporary directory; embedders running in a
//! sandbox can point it to a directory they are allowed to write to.

use std::path::PathBuf;
use std::sync::RwLock;

static TEMP_DIR: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Set the directory for temporary files, or restore the system temporary directory with `None`.
pub fn set_temp_dir(path: Option<PathBuf>) {
    *TEMP_DIR.write().unwrap_or_else(|e| e.into_inner()) = path;
}

/// The directory set with [`set_temp_dir`], or [`std::env::temp_dir`].
pub fn temp_dir() -> PathBuf {
    TEMP_DIR
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_else(std::env::temp_dir)
}
