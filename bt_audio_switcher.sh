#!/bin/bash

BLUETOOTH_DEVICE="Envaya Mini"
PROFILE_A2DP="a2dp_sink"
PROFILE_HEADSET_UNIT="handsfree_head_unit"

sink_name() {
    pacmd list-sinks  | grep '\(name:\|alias\)' | grep -B1 "${1}"  | head -1 | sed -rn 's/\s*name: <(.*?)>/\1/p'
}

card_name() {
    pacmd list-cards | grep 'name:' | grep "bluez" | sed -rn 's/\s*name: <(.*?)>/\1/p'
}

set_headset_mode() {
    CARD=$(card_name)
    if [ -z "$CARD" ]; then
        echo "Bluetooth card not found."
        exit 1
    fi
    echo "Переключаю в режим гарнитуры (микрофон + низкое качество)..."
    pacmd set-card-profile "$CARD" "$PROFILE_HEADSET_UNIT"
}

set_a2dp_mode() {
    CARD=$(card_name)
    if [ -z "$CARD" ]; then
        echo "Bluetooth card not found."
        exit 1
    fi
    echo "Переключаю в режим высокого качества (A2DP)..."
    pacmd set-card-profile "$CARD" "$PROFILE_A2DP"
}

case "$1" in
    start_record)
        set_headset_mode
        ;;
    stop_record)
        set_a2dp_mode
        ;;
    *)
        echo "Использование: $0 {start_record|stop_record}"
        ;;
esac
