#!/bin/bash

# Default values
TEMP=0.8
PREFIX="Clone the voice in the provided audio prompt."
SUFFIX="You are a helpful and energetic assistant. Always speak in English. Keep your responses concise and natural."
REF_VOICE="/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/cpp_server/assets/default_ref_audio.wav"
BRIDGE_URL="http://127.0.0.1:8090/omni/init_sys_prompt"

# Help message
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -t, --temp VALUE      Set temperature (default: $TEMP)"
  echo "  -p, --prefix TEXT     Set system prompt prefix"
  echo "  -s, --suffix TEXT     Set system prompt suffix"
  echo "  -v, --voice PATH      Path to reference audio wav"
  echo "  -h, --help            Show this help"
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--temp) TEMP="$2"; shift 2 ;;
    -p|--prefix) PREFIX="$2"; shift 2 ;;
    -s|--suffix) SUFFIX="$2"; shift 2 ;;
    -v|--voice) REF_VOICE="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# Construct JSON payload
PAYLOAD=$(cat <<EOF
{
  "media_type": "omni",
  "system_prompt_prefix": "$PREFIX",
  "system_prompt_suffix": "$SUFFIX",
  "temperature": $TEMP,
  "voice_audio": "$REF_VOICE"
}
EOF
)

echo "Setting engine config..."
echo "Temp: $TEMP"
echo "Voice: $REF_VOICE"

# Send request
curl -X POST "$BRIDGE_URL" \
     -H "Content-Type: application/json" \
     -d "$PAYLOAD"

echo -e "\nDone."
