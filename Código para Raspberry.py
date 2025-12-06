import socket
from gpiozero import PWMOutputDevice, DigitalOutputDevice
import time

AIN1 = DigitalOutputDevice(5)
AIN2 = DigitalOutputDevice(6)
PWMA = PWMOutputDevice(12)

BIN1 = DigitalOutputDevice(20)
BIN2 = DigitalOutputDevice(21)
PWMB = PWMOutputDevice(13)

STBY = DigitalOutputDevice(16)
STBY.on()

def set_motor_speeds(v_left, v_right):
    if v_left >= 0:
        AIN1.on(); AIN2.off()
    else:
        AIN1.off(); AIN2.on()
    PWMA.value = min(abs(v_left), 255) / 255.0

    if v_right >= 0:
        BIN1.on(); BIN2.off()
    else:
        BIN1.off(); BIN2.on()
    PWMB.value = min(abs(v_right), 255) / 255.0

UDP_IP = "0.0.0.0"
UDP_PORT = 5555
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Esperando comandos en puerto {UDP_PORT}...")

try:
    while True:
        data, addr = sock.recvfrom(1024)
        comando = data.decode().strip()
        print(f"Recibido: {comando}")

        if "," in comando:
            partes = comando.split(",")
            accion = partes[0].strip()
        else:
            accion = comando.strip()

        print(f"Accion Interpretada: {accion}")

        if accion == "FRENTE":
            v_left, v_right = 255, 255
        elif accion == "FRENTE_DERECHA":
            v_left, v_right = 150, 255
        elif accion == "DERECHA":
            v_left, v_right = 120, 255
        elif accion == "ATRAS_DERECHA":
            v_left, v_right = -120, -180
        elif accion == "ATRAS":
            v_left, v_right = -180, -180
        elif accion == "ATRAS_IZQUIERDA":
            v_left, v_right = -180, -120
        elif accion == "IZQUIERDA":
            v_left, v_right = 255, 120
        elif accion == "FRENTE_IZQUIERDA":
            v_left, v_right = 255, 150
        elif accion == "TROMPO":
            v_left, v_right = 255, -255
        else:
            print("Accion desconocida:", accion)
            v_left, v_right = 0, 0

        print(f"PWM aplicado: v_left={v_left}, v_right={v_right}")
        set_motor_speeds(int(v_left), int(v_right))

except KeyboardInterrupt:
    print("\nPrograma finalizado.")
    AIN1.off(); AIN2.off()
    BIN1.off(); BIN2.off()
    PWMA.value = 0
    PWMB.value = 0