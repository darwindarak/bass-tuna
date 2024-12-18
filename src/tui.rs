use cpal::traits::{DeviceTrait, HostTrait};
use lib::audio::start_audio_processing;
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

fn identify_note(f: f32, fundamental_frequencies: &[f32], n_harmonics: usize) -> (usize, f32) {
    let mut closest_index = 0;
    let mut smallest_difference = f32::MAX;
    let mut harmonic = 1;

    for (i, f_ref) in fundamental_frequencies.iter().enumerate() {
        for n in 1..=n_harmonics {
            let delta = (f - (n as f32) * f_ref).abs();
            if smallest_difference > delta {
                smallest_difference = delta;
                closest_index = i;
                harmonic = n;
            }
        }
    }
    return (closest_index, f * harmonic as f32);
}

/// Draw the pitch display screen
fn draw_pitch_display(frame: &mut Frame, model: &Model) {
    let bass_strings = ["E", "A", "D", "G"];
    let bass_frequencies = [41.2, 55.0, 73.4, 98.0];

    let pitch_text = if let Some(freq) = model.pitch {
        // Look for the closest matching pitch
        let (i, pitch) = identify_note(freq, &bass_frequencies, 3);
        let string = bass_strings[i];
        let f_ref = bass_frequencies[i];
        let cents = 1200.0 * (f_ref / pitch).log2();
        format!("Pitch {}: {:.2} Hz ({:+.1} cents)", string, pitch, cents)
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
