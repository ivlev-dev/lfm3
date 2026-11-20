use unicode_width::UnicodeWidthChar;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const ITALIC: &str = "\x1b[3m";
const DIM: &str = "\x1b[38;5;245m";
const CODE: &str = "\x1b[38;5;117m";
const HEAD: &str = "\x1b[1;38;5;81m";
const LINK: &str = "\x1b[4;38;5;111m";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Style {
    Plain,
    Bold,
    Italic,
    BoldItalic,
    Code,
    Link,
}

fn width_of(ch: char) -> usize {
    UnicodeWidthChar::width(ch).unwrap_or(0)
}

fn visible(text: &str) -> usize {
    text.chars().map(width_of).sum()
}

fn inline_runs(text: &str) -> Vec<(String, Style)> {
    let mut runs = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut buf = String::new();
    let mut i = 0;
    let flush = |buf: &mut String, runs: &mut Vec<(String, Style)>| {
        if !buf.is_empty() {
            runs.push((std::mem::take(buf), Style::Plain));
        }
    };
    while i < chars.len() {
        let rest: String = chars[i..].iter().take(3).collect();
        if rest.starts_with("***") {
            flush(&mut buf, &mut runs);
            let mut j = i + 3;
            let mut inner = String::new();
            let mut closed = false;
            while j + 2 < chars.len() {
                if chars[j] == '*' && chars[j + 1] == '*' && chars[j + 2] == '*' {
                    closed = true;
                    break;
                }
                inner.push(chars[j]);
                j += 1;
            }
            if closed {
                runs.push((inner, Style::BoldItalic));
                i = j + 3;
            } else {
                buf.push('*');
                i += 1;
            }
            continue;
        }
        if rest.starts_with("**") && chars.len() > i + 1 {
            flush(&mut buf, &mut runs);
            let mut j = i + 2;
            let mut inner = String::new();
            let mut closed = false;
            while j + 1 < chars.len() {
                if chars[j] == '*' && chars[j + 1] == '*' {
                    closed = true;
                    break;
                }
                inner.push(chars[j]);
                j += 1;
            }
            if closed {
                runs.push((inner, Style::Bold));
                i = j + 2;
            } else {
                buf.push('*');
                i += 1;
            }
            continue;
        }
        if chars[i] == '`' {
            flush(&mut buf, &mut runs);
            let mut j = i + 1;
            let mut inner = String::new();
            let mut closed = false;
            while j < chars.len() {
                if chars[j] == '`' {
                    closed = true;
                    break;
                }
                inner.push(chars[j]);
                j += 1;
            }
            if closed {
                runs.push((inner, Style::Code));
                i = j + 1;
            } else {
                buf.push('`');
                i += 1;
            }
            continue;
        }
        if chars[i] == '[' {
            let mut j = i + 1;
            let mut label = String::new();
            let mut closed = false;
            while j < chars.len() {
                if chars[j] == ']' {
                    closed = true;
                    break;
                }
                label.push(chars[j]);
                j += 1;
            }
            if closed && j + 1 < chars.len() && chars[j + 1] == '(' {
                let mut k = j + 2;
                let mut url = String::new();
                let mut done = false;
                while k < chars.len() {
                    if chars[k] == ')' {
                        done = true;
                        break;
                    }
                    url.push(chars[k]);
                    k += 1;
                }
                if done {
                    flush(&mut buf, &mut runs);
                    runs.push((label, Style::Link));
                    runs.push((format!(" ({url})"), Style::Plain));
                    i = k + 1;
                    continue;
                }
            }
        }
        if chars[i] == '*' && !(i + 1 < chars.len() && chars[i + 1] == '*') {
            let mut j = i + 1;
            let mut inner = String::new();
            let mut closed = false;
            while j < chars.len() {
                if chars[j] == '*' {
                    closed = true;
                    break;
                }
                inner.push(chars[j]);
                j += 1;
            }
            if closed && !inner.is_empty() {
                flush(&mut buf, &mut runs);
                runs.push((inner, Style::Italic));
                i = j + 1;
                continue;
            }
        }
        buf.push(chars[i]);
        i += 1;
    }
    flush(&mut buf, &mut runs);
    runs
}

fn style_prefix(style: Style) -> &'static str {
    match style {
        Style::Plain => "",
        Style::Bold => BOLD,
        Style::Italic => ITALIC,
        Style::BoldItalic => "\x1b[1;3m",
        Style::Code => CODE,
        Style::Link => LINK,
    }
}

fn wrap_runs(runs: &[(String, Style)], width: usize) -> Vec<String> {
    let width = width.max(4);
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut used = 0usize;
    for (text, style) in runs {
        for ch in text.chars() {
            let w = width_of(ch);
            if used + w > width {
                if !line.is_empty() {
                    lines.push(std::mem::take(&mut line));
                }
                used = 0;
            }
            if !line.is_empty() || *style != Style::Plain {
                line.push_str(style_prefix(*style));
            }
            line.push(ch);
            if *style != Style::Plain {
                line.push_str(RESET);
            }
            used += w;
        }
    }
    if !line.is_empty() || lines.is_empty() {
        lines.push(line);
    }
    lines
}

fn wrap_plain(text: &str, width: usize) -> Vec<String> {
    wrap_runs(&[(text.to_string(), Style::Plain)], width)
}

fn list_item(line: &str) -> Option<(String, &str)> {
    let trimmed = line.trim_start();
    let indent = line.len() - trimmed.len();
    if let Some(rest) = trimmed.strip_prefix("- ") {
        return Some((format!("{}• ", " ".repeat(indent)), rest));
    }
    if let Some(rest) = trimmed.strip_prefix("* ") {
        return Some((format!("{}• ", " ".repeat(indent)), rest));
    }
    let digits: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
    if !digits.is_empty() {
        let rest = &trimmed[digits.len()..];
        if let Some(body) = rest.strip_prefix(". ") {
            return Some((format!("{}{}. ", " ".repeat(indent), digits), body));
        }
    }
    None
}

pub fn render(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_code = false;
    for raw in text.split('\n') {
        let line = raw.trim_end();
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            if in_code {
                out.push(format!("{DIM}  └────{RESET}"));
                in_code = false;
            } else {
                let lang = trimmed.trim_start_matches('`').trim();
                let label = if lang.is_empty() { "code" } else { lang };
                out.push(format!("{DIM}  ┌─ {label}{RESET}"));
                in_code = true;
            }
            continue;
        }
        if in_code {
            for l in wrap_plain(line, width.saturating_sub(3)) {
                out.push(format!("  \x1b[48;5;236m\x1b[38;5;252m {l} {RESET}"));
            }
            continue;
        }
        if let Some(head) = trimmed.strip_prefix('#') {
            let title = head.trim_start_matches('#').trim();
            for l in wrap_runs(&[(title.to_string(), Style::Bold)], width) {
                out.push(format!("{HEAD}{l}{RESET}"));
            }
            continue;
        }
        if let Some((prefix, body)) = list_item(line) {
            let pad = visible(&prefix);
            let runs = inline_runs(body);
            for (i, l) in wrap_runs(&runs, width.saturating_sub(pad))
                .into_iter()
                .enumerate()
            {
                if i == 0 {
                    out.push(format!("{prefix}{l}"));
                } else {
                    out.push(format!("{}{l}", " ".repeat(pad)));
                }
            }
            continue;
        }
        if trimmed.is_empty() {
            out.push(String::new());
            continue;
        }
        let runs = inline_runs(line);
        for l in wrap_runs(&runs, width) {
            out.push(l);
        }
    }
    if in_code {
        out.push(format!("{DIM}  └────{RESET}"));
    }
    out
}

pub fn plain(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.split('\n') {
        for l in wrap_plain(raw, width) {
            out.push(format!("{DIM}{l}{RESET}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(line: &str) -> String {
        let mut out = String::new();
        let mut chars = line.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for n in chars.by_ref() {
                    if n.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn wraps_respect_width() {
        let lines = render("one two three four five six seven", 12);
        for l in &lines {
            assert!(visible(&strip(l)) <= 12, "line {:?}", strip(l));
        }
    }

    #[test]
    fn code_fence_is_boxed() {
        let lines = render("```rust\nlet x = 1;\n```", 40);
        assert!(lines.iter().any(|l| strip(l).contains("let x = 1;")));
        assert!(lines.first().unwrap().contains("rust"));
    }

    #[test]
    fn inline_styles_are_removed_from_text() {
        let lines = render("**bold** and `code`", 40);
        let text = strip(&lines[0]);
        assert_eq!(text, "bold and code");
    }

    #[test]
    fn lists_get_bullets() {
        let lines = render("- first\n- second", 40);
        assert_eq!(strip(&lines[0]).trim_start(), "• first");
    }

    #[test]
    fn bold_italic_triple_markers() {
        let lines = render("***both***", 40);
        assert_eq!(strip(&lines[0]), "both");
    }

    #[test]
    fn code_keeps_markers_literal() {
        let lines = render("`**x**`", 40);
        assert_eq!(strip(&lines[0]), "**x**");
    }

    #[test]
    fn headings_are_plain_text() {
        let lines = render("## Title", 40);
        assert_eq!(strip(&lines[0]), "Title");
    }
}
