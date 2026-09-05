use std::io::{self, Write};

pub struct Formatter<'w, W: Write + ?Sized> {
    writer: &'w mut W,
}

impl<'w, W: Write + ?Sized> Formatter<'w, W> {
    pub fn new(writer: &'w mut W) -> Self {
        Self { writer }
    }

    pub fn id_name(&mut self, name: &str) -> io::Result<()> {
        self.line(format_args!("id name {name}"))
    }

    pub fn id_author(&mut self, author: &str) -> io::Result<()> {
        self.line(format_args!("id author {author}"))
    }

    // There is no `option name ...` renderer: no build advertises a runtime
    // option, so the `usi` reply is identity plus `usiok` everywhere.

    pub fn usiok(&mut self) -> io::Result<()> {
        self.line(format_args!("usiok"))
    }

    pub fn readyok(&mut self) -> io::Result<()> {
        self.line(format_args!("readyok"))
    }

    pub fn info_string(&mut self, msg: &str) -> io::Result<()> {
        self.line(format_args!("info string {msg}"))
    }

    /// Emit one `info string <body>` line from pre-composed format arguments.
    ///
    /// The lazy form of [`Self::info_string`]: the caller passes `format_args!`
    /// instead of a `String`, so an interpolated message costs no allocation —
    /// and a build whose sink drops the line (the `verbose1` gate in
    /// [`crate::driver`]) never formats it at all.
    pub fn info_string_fmt(&mut self, body: std::fmt::Arguments<'_>) -> io::Result<()> {
        self.line(format_args!("info string {body}"))
    }

    /// Emit a generic `info <body>` line. The caller composes everything after
    /// the `info ` keyword (e.g. `depth 1 score cp 12 nodes 30 pv 7g7f`); the
    /// search-progress reports the driver relays go through here.
    pub fn info(&mut self, body: &str) -> io::Result<()> {
        self.line(format_args!("info {body}"))
    }

    pub fn bestmove(&mut self, move_str: &str) -> io::Result<()> {
        self.line(format_args!("bestmove {move_str}"))
    }

    /// Emit a verbatim line with no USI keyword prefix — the `isready`
    /// keep-alive's bare newline (`engine.cpp`), routed through the single
    /// output sink like every other line.
    pub fn raw_line(&mut self, text: &str) -> io::Result<()> {
        self.line(format_args!("{text}"))
    }

    fn line(&mut self, args: std::fmt::Arguments<'_>) -> io::Result<()> {
        self.writer.write_fmt(args)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn captured<F>(f: F) -> String
    where
        F: FnOnce(&mut Formatter<'_, Vec<u8>>),
    {
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut fmtr = Formatter::new(&mut buf);
            f(&mut fmtr);
        }
        String::from_utf8(buf).expect("utf-8")
    }

    #[test]
    fn id_name_emits_one_line() {
        let s = captured(|f| f.id_name("Yorkie 3.1.0").unwrap());
        assert_eq!(s, "id name Yorkie 3.1.0\n");
    }

    #[test]
    fn id_author_emits_one_line() {
        let s = captured(|f| f.id_author("Kei Ishida <ishida.kei@gmail.com>").unwrap());
        assert_eq!(s, "id author Kei Ishida <ishida.kei@gmail.com>\n");
    }

    #[test]
    fn usiok_and_readyok() {
        assert_eq!(captured(|f| f.usiok().unwrap()), "usiok\n");
        assert_eq!(captured(|f| f.readyok().unwrap()), "readyok\n");
    }

    #[test]
    fn info_string_format() {
        let s = captured(|f| f.info_string("unknown command: foo").unwrap());
        assert_eq!(s, "info string unknown command: foo\n");
    }

    #[test]
    fn info_body_format() {
        let s = captured(|f| f.info("depth 1 score cp 12 nodes 30 pv 7g7f").unwrap());
        assert_eq!(s, "info depth 1 score cp 12 nodes 30 pv 7g7f\n");
    }

    #[test]
    fn bestmove_format() {
        assert_eq!(captured(|f| f.bestmove("7g7f").unwrap()), "bestmove 7g7f\n");
        assert_eq!(
            captured(|f| f.bestmove("resign").unwrap()),
            "bestmove resign\n"
        );
    }
}
