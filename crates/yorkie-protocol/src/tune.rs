//! The search parameters of a `tune` build, read from the environment at
//! startup.

use std::io::{self, Write};
use std::os::unix::ffi::OsStrExt as _;

use yorkie_state::text::TextWriter;

use crate::formatter::Formatter;

/// The environment variable that carries the tuning list: `name=value` items
/// joined by commas. Unset or empty, every parameter keeps its default.
pub const TUNE_VAR: &str = "YORKIE_TUNE";

/// Room for the refusal notice. The widest quotes one item of the list, which
/// the operating system already bounds.
const NOTICE_BYTES: usize = 8 * 1024;

/// Set the search parameters from [`TUNE_VAR`]. A list that cannot be honoured
/// is refused whole: one `info string` saying why goes to `out`, and the result
/// is `Ok(false)`, on which the caller stops before the session starts, so no
/// `readyok` is ever sent with parameters other than the ones asked for.
///
/// Called on the main thread before anything else runs, which is what the
/// parameters' statics need of the one write they get.
pub fn apply_env(out: &mut impl Write) -> io::Result<bool> {
    let Some(list) = std::env::var_os(TUNE_VAR) else {
        return Ok(true);
    };
    match yorkie_search::params::apply(list.as_bytes()) {
        Ok(()) => Ok(true),
        Err(refusal) => {
            let mut bytes = [0u8; NOTICE_BYTES];
            let mut notice = TextWriter::new(&mut bytes);
            notice.bytes(TUNE_VAR.as_bytes()).bytes(b" refused: ");
            refusal.write_message(&mut notice);
            Formatter::new(out).info_string(notice.as_bytes())?;
            Ok(false)
        }
    }
}
