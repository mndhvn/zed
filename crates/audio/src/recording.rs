use anyhow::{Context as _, Result, anyhow};
use cpal::DeviceId;
use rodio::{Source as _, buffer::SamplesBuffer};
use std::{
    io::Cursor,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

/// A microphone recording that is captured on a dedicated thread.
///
/// Dropping this value stops capture. Call [`Self::finish`] to wait for the
/// capture thread and encode the samples as a WAV upload.
pub struct MicrophoneRecording {
    stop: Arc<AtomicBool>,
    capture_thread: Option<JoinHandle<Result<RecordedAudio>>>,
}

pub struct RecordedAudio {
    pub wav_bytes: Vec<u8>,
    pub duration: Duration,
}

impl MicrophoneRecording {
    pub fn start(input_device: Option<DeviceId>) -> Result<Self> {
        let mut microphone = crate::open_input_stream(input_device)?;
        let channels = microphone.channels();
        let sample_rate = microphone.sample_rate();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_capture = stop.clone();

        let capture_thread = thread::Builder::new()
            .name("voice-transcription-recorder".to_string())
            .spawn(move || {
                let mut samples = Vec::new();
                while !stop_capture.load(Ordering::Acquire) {
                    let Some(sample) = microphone.next() else {
                        break;
                    };
                    samples.push(sample);
                }

                if samples.is_empty() {
                    return Err(anyhow!("No microphone audio was recorded"));
                }

                let duration = Duration::from_secs_f64(
                    samples.len() as f64 / f64::from(channels.get()) / f64::from(sample_rate.get()),
                );
                let wav_bytes = encode_wav(samples, channels, sample_rate)?;
                Ok(RecordedAudio {
                    wav_bytes,
                    duration,
                })
            })
            .context("failed to start microphone capture thread")?;

        Ok(Self {
            stop,
            capture_thread: Some(capture_thread),
        })
    }

    pub fn finish(mut self) -> Result<RecordedAudio> {
        self.stop.store(true, Ordering::Release);
        self.capture_thread
            .take()
            .expect("capture thread is present until finish")
            .join()
            .map_err(|_| anyhow!("microphone capture thread panicked"))?
    }
}

impl Drop for MicrophoneRecording {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}

fn encode_wav(
    samples: Vec<rodio::Sample>,
    channels: rodio::ChannelCount,
    sample_rate: rodio::SampleRate,
) -> Result<Vec<u8>> {
    let source = SamplesBuffer::new(channels, sample_rate, samples);
    let mut output = Cursor::new(Vec::new());
    rodio::wav_to_writer(source, &mut output).context("failed to encode microphone audio")?;
    Ok(output.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rodio::nz;

    #[test]
    fn encodes_wav_with_expected_header() {
        let bytes = encode_wav(vec![0.0, 0.25, -0.25, 0.0], nz!(1), nz!(16_000)).unwrap();

        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert!(bytes.len() > 44);
    }
}
