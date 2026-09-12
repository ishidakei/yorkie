use std::io::{self, Write};

/// The USI line writer: every line this engine puts on the wire goes out
/// through one of these methods.
///
/// A line is bytes — USI is ASCII by specification — so a caller hands over the
/// bytes it composed and nothing here formats, validates or copies them.
pub struct Formatter<'w, W: Write + ?Sized> {
    writer: &'w mut W,
}

impl<'w, W: Write + ?Sized> Formatter<'w, W> {
    pub fn new(writer: &'w mut W) -> Self {
        Self { writer }
    }

    pub fn id_name(&mut self, name: &[u8]) -> io::Result<()> {
        self.line(&[b"id name ", name])
    }

    pub fn id_author(&mut self, author: &[u8]) -> io::Result<()> {
        self.line(&[b"id author ", author])
    }

    // There is no `option name ...` renderer: no build advertises a runtime
    // option, so the `usi` reply is identity plus `usiok` everywhere.

    pub fn usiok(&mut self) -> io::Result<()> {
        self.line(&[b"usiok"])
    }

    pub fn readyok(&mut self) -> io::Result<()> {
        self.line(&[b"readyok"])
    }

    pub fn info_string(&mut self, msg: &[u8]) -> io::Result<()> {
        self.line(&[b"info string ", msg])
    }

    /// Emit one `info string <parts…>` line, the parts joined with nothing
    /// between them.
    ///
    /// The lazy form of [`Self::info_string`]: a message whose pieces are a
    /// literal, a token borrowed from the command line and a rendered number
    /// goes out as those pieces, so nothing has to gather them into a buffer
    /// first — and a token as long as the whole line is written whole rather
    /// than truncated to fit one.
    ///
    /// Every such message is a diagnostic, so this exists only with their
    /// feature.
    #[cfg(feature = "verbose1")]
    pub fn info_string_parts(&mut self, parts: &[&[u8]]) -> io::Result<()> {
        self.writer.write_all(b"info string ")?;
        for part in parts {
            self.writer.write_all(part)?;
        }
        self.writer.write_all(b"\n")?;
        self.writer.flush()
    }

    /// Emit a generic `info <body>` line. The caller composes everything after
    /// the `info ` keyword (e.g. `depth 1 score cp 12 nodes 30 pv 7g7f`); the
    /// search-progress reports the session relays go through here.
    ///
    /// Those reports are the `verbose2` surface, and nothing else emits a bare
    /// `info` line, so this exists only with that feature.
    #[cfg(feature = "verbose2")]
    pub fn info(&mut self, body: &[u8]) -> io::Result<()> {
        self.line(&[b"info ", body])
    }

    pub fn bestmove(&mut self, payload: &[u8]) -> io::Result<()> {
        self.line(&[b"bestmove ", payload])
    }

    /// Emit one already-composed line — the per-reply statistics line, whose
    /// caller rendered it into a stack buffer precisely so that reporting an
    /// allocation count does not itself allocate. The bytes go to the writer as
    /// they are, with no keyword in front of them.
    ///
    /// That line is the only caller, so this exists only with its feature.
    #[cfg(feature = "verbose1")]
    pub fn composed_line(&mut self, text: &[u8]) -> io::Result<()> {
        self.line(&[text])
    }

    /// Emit a verbatim line with no USI keyword prefix — the `isready`
    /// keep-alive's bare newline, routed through the single output sink like
    /// every other line.
    pub fn raw_line(&mut self, text: &[u8]) -> io::Result<()> {
        self.line(&[text])
    }

    fn line(&mut self, parts: &[&[u8]]) -> io::Result<()> {
        for part in parts {
            self.writer.write_all(part)?;
        }
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
        let s = captured(|f| f.id_name(b"Yorkie 3.1.0").unwrap());
        assert_eq!(s, "id name Yorkie 3.1.0\n");
    }

    #[test]
    fn id_author_emits_one_line() {
        let s = captured(|f| f.id_author(b"Kei Ishida <ishida.kei@gmail.com>").unwrap());
        assert_eq!(s, "id author Kei Ishida <ishida.kei@gmail.com>\n");
    }

    #[test]
    fn usiok_and_readyok() {
        assert_eq!(captured(|f| f.usiok().unwrap()), "usiok\n");
        assert_eq!(captured(|f| f.readyok().unwrap()), "readyok\n");
    }

    #[test]
    fn info_string_format() {
        let s = captured(|f| f.info_string(b"unknown command: foo").unwrap());
        assert_eq!(s, "info string unknown command: foo\n");
    }

    #[cfg(feature = "verbose1")]
    #[test]
    fn info_string_parts_are_written_in_order_with_nothing_between() {
        let s = captured(|f| {
            f.info_string_parts(&[b"unknown command: ", b"foo", b" bar"])
                .unwrap()
        });
        assert_eq!(s, "info string unknown command: foo bar\n");
        // A part may carry bytes no USI command is spelled in — a token echoed
        // back from the line arrives as it came.
        let mut buf: Vec<u8> = Vec::new();
        Formatter::new(&mut buf)
            .info_string_parts(&[b"illegal move: ", b"\x82\xa0"])
            .unwrap();
        assert_eq!(buf, b"info string illegal move: \x82\xa0\n");
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn info_body_format() {
        let s = captured(|f| f.info(b"depth 1 score cp 12 nodes 30 pv 7g7f").unwrap());
        assert_eq!(s, "info depth 1 score cp 12 nodes 30 pv 7g7f\n");
    }

    #[cfg(feature = "verbose1")]
    #[test]
    fn composed_line_is_written_verbatim() {
        let s = captured(|f| f.composed_line(b"info string stats alloc=12").unwrap());
        assert_eq!(s, "info string stats alloc=12\n");
    }

    #[test]
    fn bestmove_format() {
        assert_eq!(
            captured(|f| f.bestmove(b"7g7f").unwrap()),
            "bestmove 7g7f\n"
        );
        assert_eq!(
            captured(|f| f.bestmove(b"resign").unwrap()),
            "bestmove resign\n"
        );
    }
}
