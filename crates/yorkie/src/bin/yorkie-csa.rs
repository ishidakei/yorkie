//! The engine as a client of the CSA server protocol: it reads where to connect
//! and who to log in as from the environment, then plays the games the server
//! offers until the server or a `SIGTERM` ends the session. It takes no
//! arguments and reads nothing on standard input; its record of the session
//! goes to standard error, or to the file `YORKIE_CSA_LOG` names.

use std::io::Write as _;
use std::process::ExitCode;

use yorkie_protocol::csa::{CsaEngine, CsaSettings};
use yorkie_state::TextWriter;

const USAGE: &[u8] = b"\
usage:
  yorkie-csa        # connect to the CSA server and play
environment:
  YORKIE_CSA_HOST      the server's host name or address
  YORKIE_CSA_PORT      its port
  YORKIE_CSA_ID        the login name
  YORKIE_CSA_PASSWORD  the password
  YORKIE_CSA_LOG       optional: the file the session log is appended to
";

fn main() -> ExitCode {
    let mut bytes = [0u8; 1024];
    let mut message = TextWriter::new(&mut bytes);
    let code = run(&mut message);
    if code != ExitCode::SUCCESS {
        let mut err = std::io::stderr().lock();
        let _ = err.write_all(b"error: ");
        let _ = err.write_all(message.as_bytes());
        let _ = err.write_all(b"\n");
    }
    code
}

fn run(message: &mut TextWriter<'_>) -> ExitCode {
    if std::env::args_os().len() > 1 {
        message
            .bytes(b"yorkie-csa takes no arguments\n")
            .bytes(USAGE);
        return ExitCode::from(2);
    }
    // SAFETY: nothing has started a thread yet; the engine's are started by
    // `CsaEngine::new` below.
    let settings = match unsafe { CsaSettings::from_env() } {
        Ok(settings) => settings,
        Err(e) => {
            e.write_message(message);
            message.byte(b'\n').bytes(USAGE);
            return ExitCode::from(2);
        }
    };
    let result = CsaEngine::new(settings).and_then(CsaEngine::run);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            message.bytes(b"csa session ended: ");
            match e.raw_os_error() {
                Some(errno) => message.bytes(b"errno ").i64(i64::from(errno)),
                None => message.bytes(b"see the session log"),
            };
            ExitCode::from(1)
        }
    }
}
