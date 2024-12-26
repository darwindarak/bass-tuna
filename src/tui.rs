use cpal::traits::{DeviceTrait, HostTrait};
use crossbeam_channel::{select, unbounded};
use lib::audio::{Block, InputBlock, LowpassBlock, PitchDetectorBlock, PulseDetectorBlock};
use lib::tuner::identify_note_name;
use ratatui::symbols;
use ratatui::widgets::{Axis, Chart, Dataset};
use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{self, Borders, List, ListItem, Paragraph, Widget},
    Frame,
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::thread;

pub struct RulerWidget<'a> {
    pub current_cents: Option<f32>,
    pub detected_note: Option<String>, // Detected musical note (e.g., "E2")
    pub major_cents: f32,
    pub n_major: i32,
    pub n_minor: i32,
    pub block: Option<widgets::Block<'a>>, // Optional block for borders and titles
}

fn build_ascii_map() -> HashMap<char, &'static str> {
    let mut map = HashMap::new();

    map.insert(
        '/',
        r"
      // 
     //  
    //   
   //    
  //     
",
    );

    map.insert(
        'A',
        r"
  AAA  
 A   A 
 AAAAA 
 A   A 
 A   A 
",
    );

    map.insert(
        'B',
        r"
 BBBB  
 B   B 
 BBBB  
 B   B 
 BBBB  
",
    );

    map.insert(
        'C',
        r"
  CCCC 
 C     
 C     
 C     
  CCCC 
",
    );

    map.insert(
        'D',
        r"
 DDDD  
 D   D 
 D   D 
 D   D 
 DDDD  
",
    );

    map.insert(
        'E',
        r"
 EEEEE 
 E     
 EEEE  
 E     
 EEEEE 
",
    );

    map.insert(
        'F',
        r"
 FFFFF 
 F     
 FFFF  
 F     
 F     
",
    );

    map.insert(
        'G',
        r"
  GGGG 
 G     
 G  GG 
 G   G 
  GGGG 
",
    );

    map.insert(
        '#',
        r"
   #  #  
 ####### 
  #  #   
#######  
 #  #    
",
    );

    map.insert(
        'b',
        r"
     b   
    b    
   bbbb  
  b   b  
 bbbbb   
",
    );

    map.insert(
        ' ',
        r"
       
       
       
       
       
",
    );

    map
}

impl Widget for RulerWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Draw the block (if provided)
        if let Some(block) = &self.block {
            block.render(area, buf);
        }

        let color = if let Some(cents) = self.current_cents {
            if cents.abs() < 2.0 {
                Color::Green
            } else if cents.abs() < 10.0 {
                Color::Yellow
            } else {
                Color::Red
            }
        } else {
            Color::default()
        };

        // Define the drawable area (subtract border size)
        let inner_area = match self.block {
            Some(block) => block.inner(area),
            None => area,
        };

        // Ensure we have enough height for the ruler
        if inner_area.height < 10 {
            return;
        }

        // Fixed range for the ruler
        let center_x = inner_area.width / 2;

        let ascii_map = build_ascii_map();
        let mut ascii_lines = vec![];
        //let step_per_tick = 2; // Cents per tick

        // Render block letters (if a note is detected)
        for c in self.detected_note.unwrap_or(" ".into()).chars() {
            if let Some(ascii) = ascii_map.get(&c) {
                let lines: Vec<&str> = ascii.split('\n').collect();
                if ascii_lines.is_empty() {
                    ascii_lines = vec![String::new(); lines.len()];
                }
                for (i, line) in lines.iter().enumerate() {
                    ascii_lines[i].push_str(line);
                    ascii_lines[i].push(' '); // Add spacing between characters
                }
            }
        }

        // Calculate starting row for block letters
        //let block_letter_height = ascii_lines.len();
        let block_start_y = inner_area.y;

        // Render block letters
        for (i, line) in ascii_lines.iter().enumerate() {
            let y = block_start_y + i as u16;
            if y < inner_area.y + inner_area.height {
                let x = inner_area.x + (inner_area.width.saturating_sub(line.len() as u16) / 2);
                for (j, c) in line.chars().enumerate() {
                    let x_pos = x + j as u16;
                    //if x_pos < inner_area.x + inner_area.width {
                    buf[(x_pos, y)]
                        .set_char(c)
                        .set_style(Style::default().fg(color).add_modifier(Modifier::BOLD));
                    //}
                }
            }
        }
        let letter_height = ascii_lines.len() as u16;

        let n_markers = self.n_major * self.n_minor;
        let minor_steps = self.major_cents / (self.n_minor as f32);

        let (label_range, marker_center) = if let Some(cents) = self.current_cents {
            let marker_center = ((cents / minor_steps).round() as i32).clamp(
                -(self.n_major as f32 * self.major_cents) as i32,
                (self.n_major as f32 * self.major_cents) as i32,
            );
            let label = format!("{:+.1}", cents);
            let label_offset = (label.len() / 2) as i32;
            let label_start = marker_center - label_offset;
            let label_end = label_start + label.len() as i32;
            for (i, c) in label.chars().enumerate() {
                let x = center_x as i32 + marker_center - label_offset + i as i32;
                buf[(inner_area.x + x as u16, inner_area.y + letter_height + 2)].set_char(c);
            }
            (label_start..label_end, marker_center)
        } else {
            (0..0, n_markers + 1)
        };

        for i in -n_markers..=n_markers {
            let x = (center_x as i32 + i) as u16;
            let is_major = i % self.n_major == 0;

            let (marker_color, marker) = if i == marker_center {
                (color, '🮋')
            } else {
                (Color::Gray, '┃')
            };

            if is_major {
                // Draw the top and bottom stems since they are always the drawn
                for y in [1, 3] {
                    buf[(inner_area.x + x, inner_area.y + y + letter_height)]
                        .set_char(marker)
                        .set_style(Style::default().fg(marker_color));
                }

                // The middle stem is only draw if it's not on top of the label
                if !label_range.contains(&i) {
                    buf[(inner_area.x + x, inner_area.y + 2 + letter_height)]
                        .set_char('┃')
                        .set_style(Style::default().fg(Color::Gray));
                }

                let label = format!("{:+}", (i / self.n_minor) as f32 * self.major_cents);
                let label_start = label.len() / 2;
                for (i, c) in label.chars().enumerate() {
                    let label_x = (inner_area.x + x).saturating_sub(label_start as u16) + i as u16;
                    if label_x < inner_area.x + inner_area.width {
                        buf[(label_x, inner_area.y + letter_height + 4)]
                            .set_char(c)
                            .set_style(Style::default().fg(Color::White));
                    }
                }
            } else if i == marker_center {
                for y in [1, 3] {
                    buf[(inner_area.x + x, inner_area.y + y + letter_height)]
                        .set_char('🮋')
                        .set_style(Style::default().fg(color));
                }
            } else {
                buf[(inner_area.x + x, inner_area.y + 3 + letter_height)]
                    .set_char('┊')
                    .set_style(Style::default().fg(Color::Gray));
            }
        }
    }
}

#[derive(Clone)]
pub enum AppState {
    DeviceSelection,
    PitchDisplay,
}

pub enum Message {
    SelectNextDevice,
    SelectPreviousDevice,
    ConfirmDevice,
    Exit,
}

pub struct Model {
    state: AppState,
    devices: Vec<String>,
    selected_device: usize,
    sample_rate: usize,
    pitch: Arc<Mutex<Option<f32>>>,
    energy_history: Arc<Mutex<VecDeque<(u128, f32)>>>,
}

impl Model {
    pub fn new() -> Self {
        let host = cpal::default_host();
        let devices = host
            .input_devices()
            .unwrap()
            .map(|d| d.name().unwrap_or_else(|_| "Unknown Device".to_string()))
            .collect();

        let state = AppState::DeviceSelection;

        Model {
            state,
            devices,
            selected_device: 0,
            sample_rate: 0,
            pitch: Arc::new(Mutex::new(None)),
            energy_history: Arc::new(Mutex::new(VecDeque::from(vec![(0u128, 0.0); 1000]))),
        }
    }
}

/// Update the model based on messages
pub fn update(model: &mut Model, msg: Message) {
    //, state_sender: &Sender<Message>) {
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

            let (pitch_ui_sender, pitch_ui_receiver) = unbounded();
            let (pulse_ui_sender, pulse_ui_receiver) = unbounded();
            let (lowpass_ui_sender, lowpass_ui_receiver) = unbounded();

            let mut input_block = InputBlock::new(&device_name);

            let sample_rate = input_block.config.sample_rate().0 as usize;
            let window_size = sample_rate / 10;
            let candidate_frequencies: Vec<f32> = (1..=24)
                .map(|n| 36.7081 * 2.0f32.powf(n as f32 / 12.0))
                .collect();
            let update_interval = 100;
            let mut pitch_detector = PitchDetectorBlock::new(
                candidate_frequencies,
                sample_rate,
                window_size,
                update_interval,
            );
            model.sample_rate = sample_rate;

            let mut lowpass =
                LowpassBlock::new(10.0, sample_rate, Some(move |x| x * x * sample_rate as f32));
            let mut pulse_detector = PulseDetectorBlock::new(1e-2);

            input_block.pipe_output_to(&mut pitch_detector);
            input_block.pipe_output_to(&mut lowpass);

            pitch_detector.add_output(pitch_ui_sender);

            lowpass.pipe_output_to(&mut pulse_detector);
            lowpass.add_output(lowpass_ui_sender);
            pulse_detector.add_output(pulse_ui_sender);

            pitch_detector.run();
            lowpass.run();
            pulse_detector.run();

            input_block.run();

            let energy_history = Arc::clone(&model.energy_history);
            let pitch = Arc::clone(&model.pitch);

            thread::spawn(move || loop {
                select! {
                    recv(pulse_ui_receiver) -> _ => {

                    },
                    recv(lowpass_ui_receiver) -> result => {
                        if let Ok((t, energy)) = result {
                            {
                                let mut e = energy_history.lock().unwrap();
                                e.pop_front();
                                e.push_back((t, energy.iter().sum::<f32>() / energy.len() as f32));
                            }
                        }
                    },
                    recv(pitch_ui_receiver) -> result => {
                        if let Ok(freq) = result {
                            *pitch.lock().unwrap() = freq;
                        }
                    },

                }
            });
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

    let header = Paragraph::new("Select an Input Device").block(
        widgets::Block::default()
            .borders(Borders::ALL)
            .title("Tuna"),
    );
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

    let list = List::new(items).block(
        widgets::Block::default()
            .borders(Borders::ALL)
            .title("Devices"),
    );
    frame.render_widget(list, chunks[1]);
}

/// Draw the pitch display screen
fn draw_pitch_display(frame: &mut Frame, model: &Model) {
    let pitch = { *model.pitch.lock().unwrap() };
    let data = {
        let e = model.energy_history.lock().unwrap();
        let len = e.len();
        let latest = e[len - 1].0;
        let mut max = 0.0f64;
        let mut data = Vec::with_capacity(e.len());
        for &(t, v) in e.iter() {
            if v as f64 > max {
                max = v as f64;
            }
            data.push(((latest - t) as f64 / -1e9, v as f64));
        }
        data
    };
    //println!("{:?}", data[0]);

    let (current_cents, detected_note) = if let Some(freq) = pitch {
        let (note, _, cents) = identify_note_name(freq);

        (Some(cents), Some(note))
    } else {
        (None, None)
    };

    let layout = Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([Constraint::Min(20), Constraint::Percentage(100)])
        .split(frame.area());

    let widget = RulerWidget {
        current_cents,
        detected_note,
        block: Some(
            widgets::Block::default()
                .borders(Borders::ALL)
                .title("Tuna"),
        ),
        major_cents: 5.0,
        n_major: 5,
        n_minor: 5,
    };

    frame.render_widget(widget, layout[0]);

    let dataset = Dataset::default()
        .graph_type(widgets::GraphType::Line)
        .name("\"Energy\"")
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(Color::Blue))
        .data(&data);

    let chart = Chart::new(vec![dataset])
        .x_axis(
            Axis::default()
                .title("Time")
                .bounds([-10.0, 0.0])
                .labels(
                    (-10..=0)
                        .map(|x| format!("{}", x as f32))
                        .collect::<Vec<String>>(),
                ),
        )
        .y_axis(Axis::default().bounds([0.0, 1.25]));

    frame.render_widget(chart, layout[1]);
}
