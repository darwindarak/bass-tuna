mod tui;
use crate::tui::{update, view, Message, Model};
use ratatui::crossterm::event::{self, Event, KeyCode};
use std::io::{self};
use std::time::Duration;

fn main() -> io::Result<()> {
    let mut terminal = ratatui::init();
    terminal.clear()?;
    let mut model = Model::new();

    loop {
        terminal.draw(|f| view(f, &model)).ok();

        if event::poll(Duration::from_millis(50))? {
            if let Ok(Event::Key(key_event)) = event::read() {
                match key_event.code {
                    KeyCode::Up => update(&mut model, Message::SelectPreviousDevice),
                    KeyCode::Down => update(&mut model, Message::SelectNextDevice),
                    KeyCode::Enter => update(&mut model, Message::ConfirmDevice),
                    KeyCode::Esc => update(&mut model, Message::Exit),
                    KeyCode::Char('q') => update(&mut model, Message::Exit),
                    _ => {}
                }
            }
        }
    }
}
