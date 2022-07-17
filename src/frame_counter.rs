use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

pub struct FrameCounter {
    history: VecDeque<Duration>,
    history_duration: Duration,
    max_history_length: usize,
    last_time: Instant,
    frame_number: usize,
}

#[allow(dead_code)]
impl FrameCounter {
    pub fn new(history_length: usize) -> FrameCounter {
        FrameCounter {
            history: VecDeque::with_capacity(history_length + 1),
            history_duration: Duration::new(0, 0),
            max_history_length: history_length,
            last_time: Instant::now(),
            frame_number: 0,
        }
    }

    pub fn on_new_frame(&mut self) {
        let now = Instant::now();
        let duration = now.saturating_duration_since(self.last_time);

        self.history.push_front(duration);
        self.history_duration += duration;

        while self.history.len() > self.max_history_length {
            if let Some(old) = self.history.pop_back() {
                self.history_duration -= old;
            }
        }

        self.last_time = now;
        self.frame_number = self.frame_number.wrapping_add(1);
    }

    pub fn frame_time(&self) -> Duration {
        self.history.front().copied().unwrap_or_default()
    }

    pub fn frame_rate(&self) -> f64 {
        self.history.len() as f64 / self.history_duration.as_secs_f64()
    }

    pub fn history_length(&self) -> usize {
        self.max_history_length
    }

    pub fn frame_number(&self) -> usize {
        self.frame_number
    }
}
