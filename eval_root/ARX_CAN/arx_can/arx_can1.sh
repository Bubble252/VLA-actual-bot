#!/bin/bash

source ~/.bashrc

CAN_DEVICE="${CAN_DEVICE:-/dev/arxcan1}"
CAN_INTERFACE="can1"
PREFERRED_SERIAL="${ARX_CAN_SERIAL:-206D35B44831}"

resolve_can_device() {
    if [ -e "$CAN_DEVICE" ]; then
        echo "$CAN_DEVICE"
        return 0
    fi

    for dev in /sys/class/tty/ttyACM*; do
        [ -e "$dev" ] || continue
        serial_file="$dev/device/../serial"
        if [ -r "$serial_file" ] && [ "$(cat "$serial_file")" = "$PREFERRED_SERIAL" ]; then
            basename "$dev" | sed 's#^#/dev/#'
            return 0
        fi
    done

    for fallback in /dev/ttyACM* /dev/ttyUSB*; do
        [ -e "$fallback" ] || continue
        echo "$fallback"
        return 0
    done

    return 1
}


start_can() {
    local resolved_device
    resolved_device="$(resolve_can_device)" || {
        echo "未找到 CAN 串口设备。可检查 /dev/ttyACM*、/dev/ttyUSB* 或设置 CAN_DEVICE 环境变量。"
        return 1
    }

    echo "使用 CAN 设备: $resolved_device"
    echo "启动 slcand..."
    sudo slcand -o -f -s8 "$resolved_device" "$CAN_INTERFACE"
    if [ $? -ne 0 ]; then
        echo "slcand 启动失败"
        return 1
    fi
    echo "配置 $CAN_INTERFACE 接口..."
    sudo ip link set "$CAN_INTERFACE" up
    
    if [ $? -ne 0 ]; then
        echo "启动 $CAN_INTERFACE 接口失败：RTNETLINK answers: Operation not supported"
        return 1
    fi
    echo "$CAN_INTERFACE 启动成功"
    return 0
}

check_can() {

    if ip link show "$CAN_INTERFACE" > /dev/null 2>&1; then
  
        if ip link show "$CAN_INTERFACE" | grep -q "UP"; then
            return 0
        else
            return 1  
        fi
    else
        return 2  
    fi
}

while true; do

    if check_can; then

        echo "CAN 接口 $CAN_INTERFACE 正常工作"
    else

        echo "$CAN_INTERFACE 掉线，重启中..."
        
        sudo ip link set "$CAN_INTERFACE" down 2>/dev/null || true
        sudo pkill -9 slcand  
        sleep 1  

        if ! start_can; then
            echo "重启 CAN 接口失败，请检查硬件或驱动。"

        fi
    fi


done
