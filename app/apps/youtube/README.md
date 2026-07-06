# YouTube

YouTube fetches subtitles/transcripts only.

## Responsibilities

- Accept either a YouTube video id or full YouTube URL.
- Normalize and store the video id.
- Fetch subtitles through `youtube-transcript.io`.
- Store transcript text, provider metadata, and finance usage.

This app does not download media. Future video/audio downloading should be a
separate tool, likely based on `yt-dlp`.
