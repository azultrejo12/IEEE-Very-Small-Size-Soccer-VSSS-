import cv2
import numpy as np
import math
import socket
import json
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import datetime


# =========================
# CLASE PID
# =========================

class PIDController:



    """Controlador PID para control suave de robots"""
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(-100, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None

        # ✅ Inicializar atributos para evitar errores
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0
        self.last_output = 0.0
        
    def reset(self):
        """Reinicia el controlador PID"""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def apply_ziegler_nichols(self, Ku, Pu):
        # Ajuste clásico PID
        self.kp = 0.6 * Ku
        self.ki = 1.2 * self.kp / Pu   # equivalente a kp/Ti
        self.kd = 0.075 * Ku * Pu      # equivalente a kp*Td
    
    def compute(self, error, current_time=None):
        """
        Calcula la salida del PID
        error: diferencia entre setpoint y valor actual
        current_time: tiempo actual (opcional, se usa time.time() por defecto)
        """
        if current_time is None:
            current_time = time.time()
        
        # Primera ejecución
        if self.last_time is None:
            self.last_time = current_time
            self.last_error = error
            return 0.0
        
        # Calcular dt
        dt = current_time - self.last_time
        if dt <= 0.0:
            dt = 0.01  # Evitar división por cero
        
        # Término proporcional
        p_term = self.kp * error
        
        # Término integral
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Término derivativo
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative
        
        # Calcular salida
        output = p_term + i_term + d_term
        # Término proporcional
        p_term = self.kp * error

        # Término integral
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Término derivativo
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative

        # Calcular salida
        output = p_term + i_term + d_term

        # Guardar últimos valores para análisis
        self.last_p = p_term
        self.last_i = i_term
        self.last_d = d_term
        self.last_output = output

        # Aplicar límites
        output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        # Actualizar para próxima iteración
        self.last_error = error
        self.last_time = current_time
        
        return output

# =========================
# FASE 1: CALIBRACIÓN DE COLORES CON CLICKS
# =========================

def calibrar_colores():
    """Calibra los rangos HSV para cada color con clicks del mouse"""
    
    colores_a_calibrar = [
        'amarillo',      # equipo
        'azul_marino',   # equipo rival
        'rojo',          # orientación
        'verde_claro',   # orientación
        'naranja'        # bola
    ]
    
    valores_calibrados = {}
    
    # Márgenes para el rango HSV basados en clicks
    MARGEN_H = 10
    MARGEN_S = 60
    MARGEN_V = 60
    
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return None
    
    for color in colores_a_calibrar:
        print(f"\n=== Calibrando: {color.upper()} ===")
        print("HAZ CLICK sobre el color que quieres calibrar")
        print("Puedes hacer varios clicks para mejorar la detección")
        print("Presiona 's' para guardar y continuar")
        print("Presiona 'r' para reiniciar clicks de este color")
        print("Presiona 'q' para saltar este color")
        
        hsv_samples = []  # Muestras de clicks para este color
        ventana_nombre = f"Calibración - {color}"
        
        def on_mouse_calibration(event, x, y, flags, param):
            nonlocal hsv_samples
            
            if event == cv2.EVENT_LBUTTONDOWN:
                hsv_frame = param
                h, s, v = hsv_frame[y, x]
                hsv_samples.append((int(h), int(s), int(v)))
                
                print(f"  Click #{len(hsv_samples)}: HSV({h}, {s}, {v}) en posición ({x},{y})")
                
                # Calcular rango actual
                if len(hsv_samples) > 0:
                    hs = np.array([p[0] for p in hsv_samples])
                    ss = np.array([p[1] for p in hsv_samples])
                    vs = np.array([p[2] for p in hsv_samples])
                    
                    h_min = max(0, hs.min() - MARGEN_H)
                    h_max = min(179, hs.max() + MARGEN_H)
                    s_min = max(0, ss.min() - MARGEN_S)
                    s_max = min(255, ss.max() + MARGEN_S)
                    v_min = max(0, vs.min() - MARGEN_V)
                    v_max = min(255, vs.max() + MARGEN_V)
                    
                    print(f"  Rango actual: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
        
        cv2.namedWindow(ventana_nombre)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convertir a HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Configurar callback del mouse
            cv2.setMouseCallback(ventana_nombre, on_mouse_calibration, hsv)
            
            # Si ya hay muestras, mostrar la máscara
            if len(hsv_samples) > 0:
                hs = np.array([p[0] for p in hsv_samples])
                ss = np.array([p[1] for p in hsv_samples])
                vs = np.array([p[2] for p in hsv_samples])
                
                h_min = max(0, hs.min() - MARGEN_H)
                h_max = min(179, hs.max() + MARGEN_H)
                s_min = max(0, ss.min() - MARGEN_S)
                s_max = min(255, ss.max() + MARGEN_S)
                v_min = max(0, vs.min() - MARGEN_V)
                v_max = min(255, vs.max() + MARGEN_V)
                
                # Crear máscara
                lower = np.array([h_min, s_min, v_min])
                upper = np.array([h_max, s_max, v_max])
                mask = cv2.inRange(hsv, lower, upper)
                
                # Aplicar la máscara
                result = cv2.bitwise_and(frame, frame, mask=mask)
                
                # Mostrar información
                pixels_detectados = cv2.countNonZero(mask)
                cv2.putText(frame, f"Pixels detectados: {pixels_detectados}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Clicks: {len(hsv_samples)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Calibrando: {color}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Marcar los puntos donde se hizo click
                for i, (h_sample, s_sample, v_sample) in enumerate(hsv_samples):
                    # Encontrar la posición aproximada (no guardamos x,y pero mostramos contador)
                    pass
                
                # Mostrar ventanas
                cv2.imshow(ventana_nombre, frame)
                cv2.imshow(f"Máscara - {color}", mask)
                cv2.imshow(f"Resultado - {color}", result)
            else:
                # Sin clicks aún, solo mostrar frame
                cv2.putText(frame, f"Calibrando: {color}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, "Haz CLICK sobre el color", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(ventana_nombre, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and len(hsv_samples) > 0:  # Guardar
                hs = np.array([p[0] for p in hsv_samples])
                ss = np.array([p[1] for p in hsv_samples])
                vs = np.array([p[2] for p in hsv_samples])
                
                h_min = max(0, hs.min() - MARGEN_H)
                h_max = min(179, hs.max() + MARGEN_H)
                s_min = max(0, ss.min() - MARGEN_S)
                s_max = min(255, ss.max() + MARGEN_S)
                v_min = max(0, vs.min() - MARGEN_V)
                v_max = min(255, vs.max() + MARGEN_V)
                
                valores_calibrados[color] = {
                    "lower": [int(h_min), int(s_min), int(v_min)],
                    "upper": [int(h_max), int(s_max), int(v_max)]
                }
                print(f"✓ {color}: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
                break
            elif key == ord('r'):  # Reiniciar clicks
                hsv_samples = []
                print(f"  Clicks reiniciados para {color}")
            elif key == ord('q'):  # Saltar
                print(f"⊗ {color} saltado")
                break
        
        cv2.destroyAllWindows()
    
    cap.release()
    
    # Guardar en archivo JSON
    with open('colores_calibrados.json', 'w') as f:
        json.dump(valores_calibrados, f, indent=4)
    
    print("\n✓ Calibración completada y guardada en 'colores_calibrados.json'")
    return valores_calibrados


# =========================
# FASE 2: SELECCIÓN DE CANCHA
# =========================

def seleccionar_cancha():
    """Permite seleccionar las 16 esquinas de la cancha"""
    points = []
    
    def click_event(event, x, y, flags, param):
        nonlocal points, temp_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Punto {len(points)}/16: ({x}, {y})")
            cv2.circle(temp_frame, (x, y), 5, (0, 255, 255), -1)
            if len(points) > 1:
                cv2.line(temp_frame, points[-2], points[-1], (0, 255, 0), 2)
            cv2.imshow("Selecciona cancha", temp_frame)
            
            if len(points) == 16:
                print("✓ 16 puntos seleccionados")
                cv2.line(temp_frame, points[-1], points[0], (0, 255, 0), 2)
                cv2.imshow("Selecciona cancha", temp_frame)
    
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ret, temp_frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: No se pudo capturar frame")
        return None
    
    print("\n=== SELECCIÓN DE CANCHA ===")
    print("Haz clic en 16 puntos del borde de la cancha")
    print("Presiona cualquier tecla cuando termines")
    
    cv2.imshow("Selecciona cancha", temp_frame)
    cv2.setMouseCallback("Selecciona cancha", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(points) == 16:
        cancha_pts = np.array(points, np.int32).reshape((-1, 1, 2))
        return cancha_pts
    return None

# =========================
# FASE 3: SISTEMA DE VISIÓN CON PID
# =========================

class Vision:
    def __init__(self, colores_calibrados, cancha_pts, camera_index=1):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise IOError("No se pudo abrir la cámara")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.cancha_pts = cancha_pts
        self.aliados = []
        self.rivales = []
        self.bola = None
        
        # Configuración de robots
        self.robots_config = {
            "robot1": ("azul_celeste", "verde_claro"),
            "robot2": ("rojo", "azul_celeste"),
            "robot3": ("verde_claro", "rojo")
        }
        
        self.mi_equipo = 'azul_marino'
        self.colores_aliados = ['azul_marino']
        
        # Controladores PID para cada robot
        # PID para ángulo de orientación (girar hacia la bola)
        self.pid_angular = {
            "robot1": PIDController(kp=1.5, ki=0.1, kd=0.3, output_limits=(-100, 100)),
            "robot2": PIDController(kp=1.5, ki=0.1, kd=0.3, output_limits=(-100, 100)),
            "robot3": PIDController(kp=1.5, ki=0.1, kd=0.3, output_limits=(-100, 100))
        }

        # ✅ Aplicar Ziegler-Nichols a todos los robots
        Ku = 1.2  # Ganancia última (definida experimentalmente)
        Pu = 2.5  # Periodo de oscilación (definido experimentalmente)

        for robot_id in self.pid_angular:
            self.pid_angular[robot_id].apply_ziegler_nichols(Ku, Pu)
        
        # PID para distancia (acercarse a la bola)
        self.pid_lineal = {
            "robot1": PIDController(kp=0.8, ki=0.05, kd=0.2, output_limits=(0, 100)),
            "robot2": PIDController(kp=0.8, ki=0.05, kd=0.2, output_limits=(0, 100)),
            "robot3": PIDController(kp=0.8, ki=0.05, kd=0.2, output_limits=(0, 100))
        }
        
        # Cargar colores calibrados
        self.colores_equipo = {}
        self.colores_orientacion = {}
        self.color_bola = {}
        
        if 'amarillo' in colores_calibrados:
            lower = colores_calibrados['amarillo']['lower']
            upper = colores_calibrados['amarillo']['upper']
            self.colores_equipo['amarillo'] = (
                (lower[0], lower[1], lower[2]),
                (upper[0], upper[1], upper[2]),
            )
        
        if 'azul_marino' in colores_calibrados:
            lower = colores_calibrados['azul_marino']['lower']
            upper = colores_calibrados['azul_marino']['upper'] 
            self.colores_equipo['azul_marino'] = (
                (lower[0], lower[1], lower[2]),
                (upper[0], upper[1], upper[2]),
            )
        
        colores_orient = ['rojo', 'verde_claro', 'azul_celeste', 'purpura']
        for col in colores_orient:
            if col in colores_calibrados:
                lower = colores_calibrados[col]['lower']
                upper = colores_calibrados[col]['upper']
                self.colores_orientacion[col] = (
                    (lower[0], lower[1], lower[2]),
                    (upper[0], upper[1], upper[2])
                )
        
        if 'naranja' in colores_calibrados:
            self.color_bola = {
                "lower": np.array(colores_calibrados['naranja']['lower']),
                "upper": np.array(colores_calibrados['naranja']['upper'])
            }
        
        print("✓ Vision inicializada con colores calibrados y controladores PID")
    
    def obtener_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def calcular_distancia(self, p1, p2):
        """Calcula la distancia euclidiana entre dos puntos"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def procesar_frame(self):
        global frame_id, resultados
        frame = self.obtener_frame()
        if frame is None:
            return
    def procesar_frame(self):
        global frame_id, resultados
        frame = self.obtener_frame()
        if frame is None:
            return
        current_time = time.time()
        t0 = time.time()

        # Preprocesamiento
        frame_filtered = cv2.bilateralFilter(frame, 9, 75, 75)
        hsv = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2HSV)

        # Detección de posiciones
        self.aliados, self.rivales, self.bola, self.tags = self.detectar_posiciones(hsv)
        t1 = time.time()

        # Detección de bola
        bola = self.detectar_bola(hsv)
        t2 = time.time()

        # Bucle de robots
        for tag in self.tags:
            if "front_marker" in tag and "robot_id" in tag:
                robot_id = tag["robot_id"]

                # Dibujar líneas
                cv2.line(frame, tuple(np.int32(tag["centro"])),
                        tuple(np.int32(tag["front_marker"])), (0, 255, 255), 3)
                if self.bola is not None:
                    cv2.line(frame, tuple(np.int32(tag["centro"])),
                            tuple(np.int32(self.bola)), (255, 0, 255), 2)

                # Calcular ángulo relativo
                result = self.clasificar_bola_relativa(tag["centro"], tag["front_marker"], self.bola)
                t3 = time.time()

                if result is not None:
                    angle_error, pos = result
                    distancia = self.calcular_distancia(tag["centro"], self.bola)
                    error_angular = angle_error
                    error_distancia = distancia


                    # PID
                    control_angular = self.pid_angular[robot_id].compute(angle_error, time.time())
                    control_lineal = self.pid_lineal[robot_id].compute(distancia, time.time())

                    # Comando
                    comando = self.generar_comando_pid(control_angular, control_lineal, angle_error)
                    t4 = time.time()

                    # Guardar tiempos (✅ solo aquí, no en otro lado)
                    resultados.append({
                        "frame": frame_id,
                        "tiempo_colores": t1 - t0,
                        "tiempo_bola": t2 - t1,
                        "tiempo_angulo": t3 - t2,
                        "tiempo_linea": t4 - t3,
                        "total": t4 - t0,
                        "error_angular": error_angular,
                        "error_distancia": error_distancia,
                        "pid_ang_P": self.pid_angular[robot_id].last_p,
                        "pid_ang_I": self.pid_angular[robot_id].last_i,
                        "pid_ang_D": self.pid_angular[robot_id].last_d,
                        "pid_ang_out": self.pid_angular[robot_id].last_output,
                        "pid_lin_P": self.pid_lineal[robot_id].last_p,
                        "pid_lin_I": self.pid_lineal[robot_id].last_i,
                        "pid_lin_D": self.pid_lineal[robot_id].last_d,
                        "pid_lin_out": self.pid_lineal[robot_id].last_output

                        
                    })
                    frame_id += 1

                    # Enviar comando
                    comando_final = traducir_comando(comando)
                    enviar_orden(robot_id, comando_final)
                    print(f"[CMD] {robot_id} → {comando_final}")

        cv2.imshow("Vision Sistema", frame)

        

        
        # Máscara de cancha
        if self.cancha_pts is not None:
            mask_field = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_field, [self.cancha_pts], 255)
            frame = cv2.bitwise_and(frame, frame, mask=mask_field)
        
        # Preprocesamiento mejorado
        frame_filtered = cv2.bilateralFilter(frame, 9, 75, 75)
        hsv = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2HSV)
        
        # Detectar todo
        self.aliados, self.rivales, self.bola, self.tags = self.detectar_posiciones(hsv)
        # Mapeo de colores
        color_bgr_map = {
            'amarillo': (0, 255, 255),
            'azul_marino': (255, 0, 0),
            'rojo': (0, 0, 255),
            'verde_claro': (0, 255, 0),
            'azul_celeste': (255, 255, 0),
            'purpura': (255, 0, 255)
        }
        
        # Dibujar bola
        if self.bola is not None:
            cv2.circle(frame, tuple(np.int32(self.bola)), 8, (0, 165, 255), -1)
            cv2.circle(frame, tuple(np.int32(self.bola)), 12, (0, 255, 255), 2)
        
        # Dibujar contornos
        rectangulos = self.detectar_formas(hsv, self.colores_equipo, tipo='rectangulo')
        cuadros = self.detectar_formas(hsv, self.colores_orientacion, tipo='cuadro')
        
        for nombre, _, cnt in rectangulos:
            cv2.drawContours(frame, [cnt], -1, color_bgr_map.get(nombre, (255, 255, 255)), 2)
        
        for nombre, centro, cnt in cuadros:
            cv2.drawContours(frame, [cnt], -1, color_bgr_map.get(nombre, (255, 255, 255)), 2)
            cv2.circle(frame, tuple(np.int32(centro)), 5, (255, 255, 255), -1)
        
        # Dibujar robots y aplicar control PID
        for tag in self.tags:
            if "front_marker" in tag and "robot_id" in tag:
                robot_id = tag["robot_id"]
                
                # Línea de orientación
                cv2.line(frame, tuple(np.int32(tag["centro"])),
                        tuple(np.int32(tag["front_marker"])),
                        (0, 255, 255), 3)
                
                # Línea hacia bola
                if self.bola is not None:
                    cv2.line(frame, tuple(np.int32(tag["centro"])),
                            tuple(np.int32(self.bola)),
                            (255, 0, 255), 2)
                    
                    # Calcular posición relativa
                    result = self.clasificar_bola_relativa(
                        tag["centro"], tag["front_marker"], self.bola
                    )
                    
                    if result is not None:
                        angle_error, pos = result
                        distancia = self.calcular_distancia(tag["centro"], self.bola)
                        
                        # Aplicar control PID
                        control_angular = self.pid_angular[robot_id].compute(angle_error, current_time)
                        control_lineal = self.pid_lineal[robot_id].compute(distancia, current_time)
                        
                        # Generar comando basado en PID
                        comando = self.generar_comando_pid(control_angular, control_lineal, angle_error)

                        # Mostrar info
                        text = f"{robot_id}: {pos} | A:{angle_error:.1f}° D:{distancia:.0f}px"
                        y_pos = 40 + 35 * int(robot_id[-1])
                        cv2.putText(frame, text, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Mostrar valores PID
                        text_pid = f"  PID: Ang={control_angular:.1f} Lin={control_lineal:.1f}"
                        cv2.putText(frame, text_pid, (10, y_pos + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        # Dibujar líneas de orientación y hacia la bola
                        cv2.line(frame, tuple(np.int32(tag["centro"])),
                            tuple(np.int32(tag["front_marker"])),
                            (0, 255, 255), 3)
                        if self.bola is not None:
                            cv2.line(frame, tuple(np.int32(tag["centro"])),
                                tuple(np.int32(self.bola)),
                                (255, 0, 255), 2)
                        # Enviar comando
                        comando_final = traducir_comando(comando)
                        enviar_orden(robot_id, comando_final)
                        print(f"[CMD] {robot_id} → {comando_final}")
                

        
        cv2.imshow("Vision Sistema", frame)
    
    def generar_comando_pid(self, control_angular, control_lineal, angle_error):
        """
        Genera comando para el robot basado en salidas PID
        control_angular: salida del PID angular (-100 a 100)
        control_lineal: salida del PID lineal (0 a 100)
        angle_error: error angular en grados
        """
        # Normalizar ángulo de error
        abs_angle = abs(angle_error)
        
        # Si el error angular es grande, priorizar rotación
        if abs_angle > 45:
            if control_angular > 0:
                return "GIRAR_DERECHA"
            else:
                return "GIRAR_IZQUIERDA"
        
        # Si el error angular es moderado
        elif abs_angle > 15:
            if control_angular > 0:
                return "FRENTE_DERECHA"
            else:
                return "FRENTE_IZQUIERDA"
        
        # Si está bien alineado, avanzar
        else:
            # Modular velocidad según control lineal
            if control_lineal > 70:
                return "AVANZAR_RAPIDO"
            elif control_lineal > 30:
                return "AVANZAR"
            elif control_lineal > 10:
                return "AVANZAR_LENTO"
            else:
                return "DETENERSE"


    def detectar_posiciones(self, frame):
        aliados_dict = {}
        rivales = []
        bola = self.detectar_bola(frame)
        tags_detectadas = self.detectar_tags(frame)
        
        for tag in tags_detectadas:
            if tag["color"] in self.colores_aliados and "robot_id" in tag:
                aliados_dict[tag["robot_id"]] = {"centro": tag["centro"], "angulo": tag["angulo"]}
            else:
                rivales.append(tag["centro"])
        
        aliados = []
        for robot_id in ["robot1", "robot2", "robot3"]:
            if robot_id in aliados_dict:
                aliados.append(aliados_dict[robot_id]["centro"])
            else:
                aliados.append(None)
        
        return aliados, rivales, bola, tags_detectadas
    
    def detectar_tags(self, hsv):
        rectangulos = self.detectar_formas(hsv, self.colores_equipo, tipo='rectangulo')
        rectangulos = [r for r in rectangulos if r[0] == self.mi_equipo]
        cuadros = self.detectar_formas(hsv, self.colores_orientacion, tipo='cuadro')
        
        tags_detectados = []
        for color_rect, centro_rect, cnt_rect in rectangulos:
            cuadros_cercanos = sorted(cuadros, key=lambda c: self.distancia(centro_rect, c[1]))[:2]
            if len(cuadros_cercanos) < 2:
                continue
            
            colores_detectados = {cuadros_cercanos[0][0], cuadros_cercanos[1][0]}
            robot_id = None
            for rid, colores_robot in self.robots_config.items():
                if set(colores_robot) == colores_detectados:
                    robot_id = rid
                    break
            
            if robot_id is None:
                continue
            
            p1, p2 = cuadros_cercanos[0][1], cuadros_cercanos[1][1]
            punto_medio = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            dx, dy = punto_medio[0] - centro_rect[0], punto_medio[1] - centro_rect[1]
            angulo_deg = (math.degrees(math.atan2(dy, dx)) + 360) % 360
            
            tags_detectados.append({
                "color": color_rect,
                "centro": centro_rect,
                "angulo": angulo_deg,
                "robot_id": robot_id,
                "front_marker": punto_medio,
                "contorno": cnt_rect
            })
        
        return tags_detectados
    
    def detectar_formas(self, hsv, colores, tipo='cuadro'):
        encontrados = []
        
        for nombre, (lower, upper) in colores.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Limpieza de ruido
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contornos:
                area = cv2.contourArea(cnt)
                if tipo == 'cuadro' and area < 100:
                    continue
                if tipo == 'rectangulo' and (area < 300 or area > 20000):
                    continue
                
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                encontrados.append((nombre, (cx, cy), cnt))
        
        return encontrados
    
    def detectar_bola(self, frame_hsv):
        if not self.color_bola:
            return None
        
        mascara = cv2.inRange(frame_hsv, self.color_bola["lower"], self.color_bola["upper"])
        kernel = np.ones((5, 5), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            contorno_principal = max(contornos, key=cv2.contourArea)
            area = cv2.contourArea(contorno_principal)
            if area > 100:
                M = cv2.moments(contorno_principal)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        return None
    
    def distancia(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def angle_deg(self, v):
        return math.degrees(math.atan2(v[1], v[0]))
    
    def normalize_angle_deg(self, a):
        return (a + 180) % 360 - 180
    
    def clasificar_bola_relativa(self, car_center, front_marker, ball_center):
        if car_center is None or front_marker is None or ball_center is None:
            return None
        
        orient_vec = np.array(front_marker, dtype=float) - np.array(car_center, dtype=float)
        ball_vec = np.array(ball_center, dtype=float) - np.array(car_center, dtype=float)
        
        if np.linalg.norm(orient_vec) == 0 or np.linalg.norm(ball_vec) == 0:
            return None
        
        ang_orient = self.angle_deg(orient_vec)
        ang_ball = self.angle_deg(ball_vec)
        delta = self.normalize_angle_deg(ang_ball - ang_orient)
        
        if -22.5 <= delta <= 22.5:
            pos = 'FRENTE'
        elif 22.5 < delta <= 67.5:
            pos = 'FRENTE_DERECHA'
        elif 67.5 < delta <= 112.5:
            pos = 'DERECHA'
        elif 112.5 < delta <= 157.5:
            pos = 'ATRAS_DERECHA'
        elif delta > 157.5 or delta < -157.5:
            pos = 'ATRAS'
        elif -157.5 <= delta < -112.5:
            pos = 'ATRAS_IZQUIERDA'
        elif -112.5 <= delta < -67.5:
            pos = 'IZQUIERDA'
        elif -67.5 <= delta < -22.5:
            pos = 'FRENTE_IZQUIERDA'
        else:
            pos = 'DESCONOCIDO'
        
        return delta, pos

# =========================
# CONFIGURACIÓN DE RED
# =========================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

robots = {
    "robot1": ("192.168.3.17", 5555),
    "robot2": ("192.168.3.19", 5555),
    "robot3": ("192.168.3.18", 5555),
}
# Buffer para guardar tiempos de cada frame
resultados = []
frame_id = 0

def traducir_comando(comando_pid):
    """
    Traduce los comandos PID a los labels que entiende el carrito
    """
    mapa = {
        "AVANZAR_RAPIDO": "FRENTE",
        "AVANZAR": "FRENTE",
        "AVANZAR_LENTO": "FRENTE",
        "DETENERSE": "ATRAS",
        "GIRAR_DERECHA": "DERECHA",
        "GIRAR_IZQUIERDA": "IZQUIERDA",
        "FRENTE_DERECHA": "FRENTE_DERECHA",
        "FRENTE_IZQUIERDA": "FRENTE_IZQUIERDA"
    }
    return mapa.get(comando_pid, "DESCONOCIDO")
def enviar_orden(robot, comando):
    if robot not in robots:
        return
    try:
        sock.sendto(comando.encode(), robots[robot])
        print(f"[-] {robot}: {comando}")
    except Exception as e:
        print(f"[X] Error {robot}: {e}")


# ============================================
# MAIN: FLUJO COMPLETO
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("SISTEMA DE VISIÓN CON PID PARA ROBOCUP")
    print("=" * 50)

    # Opción 1: Calibrar colores nuevamente
    print("\n1. ¿Deseas calibrar colores? (s/n): ", end='')
    respuesta = input().strip().lower()

    colores_calibrados = None
    if respuesta == 's':
        colores_calibrados = calibrar_colores()
    else:
        if os.path.exists('colores_calibrados.json'):
            print("/ Cargando calibración existente ...")
            with open('colores_calibrados.json', 'r') as f:
                colores_calibrados = json.load(f)
        else:
            print("A No hay calibración guardada. Ejecutando calibración...")
            colores_calibrados = calibrar_colores()

    if not colores_calibrados:
        print("X Error: No se pudo obtener calibración")
        exit(1)

    # Opción 2: Seleccionar cancha
    print("\n2. ¿Deseas seleccionar la cancha? (s/n): ", end='')
    respuesta = input().strip().lower()
    cancha_pts = None
    if respuesta == 's':
        cancha_pts = seleccionar_cancha()

    # Opción 3: Ejecutar sistema de visión con PID
    print("\n" + "=" * 50)
    print("INICIANDO SISTEMA DE VISIÓN CON PID")
    print("=" * 50)
    print("Presiona 'q' para salir\n")

    vision = Vision(colores_calibrados, cancha_pts, camera_index=1)

    try:
        while True:
            vision.procesar_frame()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
            # Enviar STOP a todos los robots antes de cerrar
        for robot_id in robots.keys():
            enviar_orden(robot_id, "STOP")
        vision.cap.release()
        cv2.destroyAllWindows()
            # Exportar resultados a Excel
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        df = pd.DataFrame(resultados)
        df.to_excel(f"resultados_tiempos_zieglernichols_DAVICHO_I7{timestamp}.xlsx", index=False)
        print("✓ Datos guardados en resultados_tiempos.xlsx")

        

        # Graficar tiempos
        plt.figure(figsize=(10,6))
        plt.plot(df["frame"], df["tiempo_colores"], label="Colores")
        plt.plot(df["frame"], df["tiempo_bola"], label="Bola")
        plt.plot(df["frame"], df["tiempo_angulo"], label="Ángulo")
        plt.plot(df["frame"], df["tiempo_linea"], label="Línea óptima")
        plt.plot(df["frame"], df["total"], label="Total", linewidth=2, color="black")
        plt.legend()
        plt.xlabel("Frame")
        plt.ylabel("Tiempo (s)")
        plt.title("Tiempos de procesamiento por etapa")
        plt.savefig(f"grafica_tiempos_ziglernichols{timestamp}.png")

        plt.figure(figsize=(10,6))
        plt.plot(df["frame"], df["error_distancia"], label="error distancia")
        plt.plot(df["frame"], df["error_angular"], label="error angular")
        plt.plot(df["frame"], df["total"], label="Total", linewidth=2, color="black")
        plt.legend()
        plt.xlabel("Tiempo(s)")
        plt.ylabel("Error")
        plt.title("Error distancia/angular")
        plt.savefig(f"Error distancia_angular_ziglernichols{timestamp}.png")

        plt.figure(figsize=(10,6))
        plt.plot(df["frame"], df["pid_ang_P"], label="P angular")
        plt.plot(df["frame"], df["pid_ang_I"], label="I angular")
        plt.plot(df["frame"], df["pid_ang_D"], label="D angular")
        plt.plot(df["frame"], df["pid_ang_out"], label="out angular")
        plt.plot(df["frame"], df["total"], label="Total", linewidth=2, color="black")
        plt.legend()
        plt.xlabel("Tiempo (s)")
        plt.ylabel("PID angular")
        plt.title("PID ANGULAR")
        plt.savefig(f"PID angular_ziglernichols{timestamp}.png")


        plt.figure(figsize=(10,6))
        plt.plot(df["frame"], df["pid_lin_P"], label="P lineal")
        plt.plot(df["frame"], df["pid_lin_I"], label="I lineal")
        plt.plot(df["frame"], df["pid_lin_D"], label="D lineal")
        plt.plot(df["frame"], df["pid_lin_out"], label="out lineal")
        plt.plot(df["frame"], df["total"], label="Total", linewidth=2, color="black")
        plt.legend()
        plt.xlabel("Tiempo (s)")
        plt.ylabel("PID lineal")
        plt.title("PID LINEAL")
        plt.savefig(f"PID lineal_ziglernichols{timestamp}.png")


        # Simulación simple de respuesta al escalón
        t = np.linspace(0, 10, 200)

        # Ejemplo de tres respuestas (puedes ajustar fórmulas según tu sistema real)
        response_bad = 1 - np.exp(-t) * np.cos(2*t)   # mal ajustado
        response_zn = 1 - np.exp(-t/2) * np.cos(t)    # Ziegler-Nichols
        response_tuned = 1 - np.exp(-t/3)             # ajuste fino manual

        plt.figure(figsize=(10,6))
        plt.plot(t, response_bad, label="Mal ajustado")
        plt.plot(t, response_zn, label="Ziegler-Nichols")
        plt.plot(t, response_tuned, label="Ajuste fino")
        plt.legend()
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Salida")
        plt.title("Comparación de ajustes PID (respuesta al escalón)")
        plt.savefig(f"comparacion_PID_ziglernichols{timestamp}.png")
        plt.show()

        print("\n/ Sistema finalizado")
