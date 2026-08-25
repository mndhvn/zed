use futures::AsyncReadExt as _;
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest};
use serde::Deserialize;

use crate::RequestError;

pub const DEFAULT_TRANSCRIPTION_MODEL: &str = "gpt-transcribe";

#[derive(Deserialize)]
struct TranscriptionResponse {
    text: String,
}

/// Transcribes a WAV recording with OpenAI's file transcription endpoint.
pub async fn transcribe_wav(
    client: &dyn HttpClient,
    api_url: &str,
    api_key: &str,
    wav_bytes: Vec<u8>,
) -> Result<String, RequestError> {
    let boundary = format!("----ZedVoiceBoundary{:x}", rand::random::<u64>());
    let mut body = Vec::with_capacity(wav_bytes.len() + 512);

    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
    body.extend_from_slice(DEFAULT_TRANSCRIPTION_MODEL.as_bytes());
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"recording.wav\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body.extend_from_slice(&wav_bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(format!("{api_url}/audio/transcriptions"))
        .header("Authorization", format!("Bearer {}", api_key.trim()))
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(AsyncBody::from(body))
        .map_err(|error| RequestError::Other(error.into()))?;

    let mut response = client.send(request).await?;
    let status = response.status();
    let headers = response.headers().clone();
    let mut body = String::new();
    response
        .body_mut()
        .read_to_string(&mut body)
        .await
        .map_err(|error| RequestError::Other(error.into()))?;

    if !status.is_success() {
        return Err(RequestError::HttpResponseError {
            provider: "OpenAI transcription".to_owned(),
            status_code: status,
            body,
            headers: Box::new(headers),
        });
    }

    serde_json::from_str::<TranscriptionResponse>(&body)
        .map(|response| response.text)
        .map_err(|error| RequestError::Other(error.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use http_client::{FakeHttpClient, Response};
    use std::sync::{Arc, Mutex};

    #[test]
    fn sends_wav_as_authenticated_multipart_form() {
        let captured_request = Arc::new(Mutex::new(None));
        let captured_request_for_handler = captured_request.clone();
        let client = FakeHttpClient::create(move |mut request| {
            let captured_request = captured_request_for_handler.clone();
            async move {
                let uri = request.uri().to_string();
                let authorization = request.headers()["authorization"]
                    .to_str()
                    .unwrap()
                    .to_owned();
                let content_type = request.headers()["content-type"]
                    .to_str()
                    .unwrap()
                    .to_owned();
                let mut body = Vec::new();
                request.body_mut().read_to_end(&mut body).await?;
                captured_request
                    .lock()
                    .unwrap()
                    .replace((uri, authorization, content_type, body));

                Ok(Response::builder()
                    .status(200)
                    .body(AsyncBody::from(r#"{"text":"hello from the microphone"}"#))?)
            }
        });

        let text = block_on(transcribe_wav(
            client.as_ref(),
            "https://api.openai.com/v1",
            " secret ",
            b"RIFFtest-WAVE".to_vec(),
        ))
        .unwrap();

        assert_eq!(text, "hello from the microphone");
        let request = captured_request.lock().unwrap();
        let (uri, authorization, content_type, body) = request.as_ref().unwrap();
        assert_eq!(uri, "https://api.openai.com/v1/audio/transcriptions");
        assert_eq!(authorization, "Bearer secret");
        assert!(content_type.starts_with("multipart/form-data; boundary="));
        assert!(
            body.windows(DEFAULT_TRANSCRIPTION_MODEL.len())
                .any(|window| { window == DEFAULT_TRANSCRIPTION_MODEL.as_bytes() })
        );
        assert!(
            body.windows(b"filename=\"recording.wav\"".len())
                .any(|window| { window == b"filename=\"recording.wav\"" })
        );
        assert!(
            body.windows(b"RIFFtest-WAVE".len())
                .any(|window| { window == b"RIFFtest-WAVE" })
        );
    }
}
