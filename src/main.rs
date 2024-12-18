mod tui;
use crate::tui::{update, view, Message, Model};
use ratatui::crossterm::event::{self, Event, KeyCode};
use std::io::{self};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Duration;

fn main() -> io::Result<()> {
    let mut terminal = ratatui::init();
    terminal.clear()?;

    let (pitch_tx, pitch_rx): (Sender<Option<f32>>, Receiver<Option<f32>>) = channel();
    let mut model = Model::new();

    loop {
        terminal.draw(|f| view(f, &model)).ok();

        if event::poll(Duration::from_millis(100))? {
            if let Ok(event) = event::read() {
                match event {
                    Event::Key(key_event) => match key_event.code {
                        KeyCode::Up => update(&mut model, Message::SelectPreviousDevice, &pitch_tx),
                        KeyCode::Down => update(&mut model, Message::SelectNextDevice, &pitch_tx),
                        KeyCode::Enter => update(&mut model, Message::ConfirmDevice, &pitch_tx),
                        KeyCode::Esc => update(&mut model, Message::Exit, &pitch_tx),
                        KeyCode::Char('q') => update(&mut model, Message::Exit, &pitch_tx),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }

        if let Ok(pitch) = pitch_rx.try_recv() {
            update(&mut model, Message::UpdatePitch(pitch), &pitch_tx);
        }
    }
}
