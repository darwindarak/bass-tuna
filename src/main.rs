mod tui;
use crate::tui::{update, view, AppState, Message, Model};
use crossbeam_channel::{unbounded, Receiver, Sender};
use ratatui::crossterm::event::{self, Event, KeyCode};
use std::io::{self};
use std::time::Duration;

fn main() -> io::Result<()> {
    let mut terminal = ratatui::init();
    terminal.clear()?;

    let (state_sender, state_receiver): (Sender<AppState>, Receiver<AppState>) = unbounded();
    let mut model = Model::new();

    loop {
        terminal.draw(|f| view(f, &model)).ok();

        if event::poll(Duration::from_millis(100))? {
            if let Ok(event) = event::read() {
                match event {
                    Event::Key(key_event) => match key_event.code {
                        KeyCode::Up => {
                            update(&mut model, Message::SelectPreviousDevice, &state_sender)
                        }
                        KeyCode::Down => {
                            update(&mut model, Message::SelectNextDevice, &state_sender)
                        }
                        KeyCode::Enter => update(&mut model, Message::ConfirmDevice, &state_sender),
                        KeyCode::Esc => update(&mut model, Message::Exit, &state_sender),
                        KeyCode::Char('q') => update(&mut model, Message::Exit, &state_sender),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }

        if let Ok(state) = state_receiver.try_recv() {
            update(&mut model, Message::UpdateState(state), &state_sender);
        }
    }
}
