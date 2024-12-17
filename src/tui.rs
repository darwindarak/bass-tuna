use crate::audio::start_audio_processing;
use cpal::traits::{DeviceTrait, HostTrait};
use ratatui::{
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};
use std::sync::mpsc::Sender;

enum AppState {
    DeviceSelection,
    PitchDisplay,
}

pub enum Message {
    SelectNextDevice,
    SelectPreviousDevice,
    ConfirmDevice,
    UpdatePitch(Option<f32>),
    Exit,
}

pub struct Model {
    state: AppState,
    devices: Vec<String>,
    selected_device: usize,
    pitch: Option<f32>,
}

impl Model {
    pub fn new() -> Self {
        let host = cpal::default_host();
        let devices = host
            .input_devices()
            .unwrap()
            .map(|d| d.name().unwrap_or_else(|_| "Unknown Device".to_string()))
            .collect();

        Model {
            state: AppState::DeviceSelection,
            devices,
            selected_device: 0,
            pitch: None,
        }
    }
}

/// Update the model based on messages
pub fn update(model: &mut Model, msg: Message, pitch_tx: &Sender<Option<f32>>) {
    match msg {
        Message::SelectNextDevice => {
            if model.selected_device < model.devices.len() - 1 {
                model.selected_device += 1;
            }
        }
        Message::SelectPreviousDevice => {
            if model.selected_device > 0 {
                model.selected_device -= 1;
            }
        }
        Message::ConfirmDevice => {
            let device_name = model.devices[model.selected_device].clone();
            model.state = AppState::PitchDisplay;
            start_audio_processing(device_name, pitch_tx.clone());
        }
        Message::UpdatePitch(pitch) => {
            model.pitch = pitch;
        }
        Message::Exit => {
            ratatui::restore();
            std::process::exit(0);
        }
    }
}

/// View function to render the UI
pub fn view(frame: &mut Frame, model: &Model) {
    match model.state {
        AppState::DeviceSelection => draw_device_selection(frame, model),
        AppState::PitchDisplay => draw_pitch_display(frame, model),
    }
}

/// Draw the device selection screen
fn draw_device_selection(frame: &mut Frame, model: &Model) {
    let chunks = Layout::default()
        .constraints([Constraint::Percentage(20), Constraint::Percentage(80)])
        .split(frame.area());

    let header = Paragraph::new("Select an Input Device")
        .block(Block::default().borders(Borders::ALL).title("Audio Tuner"));
    frame.render_widget(header, chunks[0]);

    let items: Vec<ListItem> = model
        .devices
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let style = if i == model.selected_device {
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(Span::styled(d.clone(), style))
        })
        .collect();

    let list = List::new(items).block(Block::default().borders(Borders::ALL).title("Devices"));
    frame.render_widget(list, chunks[1]);
}

/// Draw the pitch display screen
fn draw_pitch_display(frame: &mut Frame, model: &Model) {
    let bass_strings = ["E", "A", "D", "G"];
    let bass_frequencies = [41.2, 55.0, 73.4, 98.0];

    let pitch_text = if let Some(freq) = model.pitch {
        // Look for the closest matching pitch
        let mut closest_choice = 0;
        let mut smallest_difference = f32::MAX;
        for (i, f_ref) in bass_frequencies.iter().enumerate() {
            let delta = (f_ref - freq).abs();
            if delta < smallest_difference {
                closest_choice = i;
                smallest_difference = delta;
            }
        }
        let string = bass_strings[closest_choice];
        let f_ref = bass_frequencies[closest_choice];
        let cents = 1200.0 * (f_ref / freq).log2();
        format!("Pitch {}: {:.2} Hz ({:+.1} cents)", string, freq, cents)
    } else {
        format!("Listening...")
    };

    let paragraph = Paragraph::new(pitch_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Pitch Detection"),
        )
        .style(Style::default().fg(Color::Cyan));

    frame.render_widget(paragraph, frame.area());
}
