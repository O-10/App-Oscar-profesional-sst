import streamlit as st
import pandas as pd
import stripe
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
import cv2
import numpy as np
from datetime import datetime

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Sistema de Detección de Riesgos Laborales",
    page_icon="⚠️",
    layout="wide"
)

# --- USUARIOS ---
users = {
    "usuario1": "clave123",
    "oscar": "segura456"
}

# --- CLAVE SECRETA STRIPE ---
# Mejor práctica: usar variables de entorno
stripe_key = os.environ.get("ZTgQTJF0CA75bTfQixhE", "sk_test_tu_clave_privada")
stripe.api_key = stripe_key

# --- CLASIFICADOR DE DETECCIÓN DE EPP ---
def cargar_clasificadores():
    try:
        # Clasificador para detectar cascos
        casco_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Se usará para simular detección de cascos
        # Clasificador para detectar chalecos (simulado)
        chaleco_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        return casco_cascade, chaleco_cascade
    except Exception as e:
        st.error(f"Error al cargar clasificadores: {e}")
        return None, None

# --- DETECCIÓN DE RIESGOS EN IMÁGENES ---
def detectar_riesgos(imagen):
    casco_cascade, chaleco_cascade = cargar_clasificadores()
    if casco_cascade is None or chaleco_cascade is None:
        return imagen, []
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Buscar cascos (simulado con detección facial)
    cascos = casco_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Buscar chalecos (simulado con detección de cuerpo)
    chalecos = chaleco_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Copiar imagen para mostrar resultados
    imagen_con_detecciones = imagen.copy()
    
    alertas = []
    
    # Mostrar detecciones de cascos
    if len(cascos) == 0:
        alertas.append("⚠️ ALERTA: No se detectaron cascos de seguridad")
    
    for (x, y, w, h) in cascos:
        cv2.rectangle(imagen_con_detecciones, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(imagen_con_detecciones, 'Casco', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Mostrar detecciones de chalecos
    if len(chalecos) == 0:
        alertas.append("⚠️ ALERTA: No se detectaron chalecos reflectantes")
        
    for (x, y, w, h) in chalecos:
        cv2.rectangle(imagen_con_detecciones, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(imagen_con_detecciones, 'Chaleco', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    # Análisis de zonas de riesgo (simulación)
    # Detectamos áreas de color rojo que podrían indicar zonas de peligro
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    mask = mask1 + mask2
    
    # Contar píxeles rojos para determinar zonas de peligro
    red_pixels = np.sum(mask > 0)
    if red_pixels > imagen.shape[0] * imagen.shape[1] * 0.05:  # Si hay más de 5% de píxeles rojos
        alertas.append("🚨 ALERTA: Posible zona de peligro detectada")
        # Resaltar zonas rojas
        red_zone = cv2.bitwise_and(imagen, imagen, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filtrar contornos pequeños
                cv2.drawContours(imagen_con_detecciones, [contour], -1, (0, 0, 255), 3)
    
    return imagen_con_detecciones, alertas

# --- ANÁLISIS DE RIESGOS POR ÁREA ---
def calcular_nivel_riesgo(area, num_alertas):
    base_risk = {
        "Construcción": 5,
        "Bodega": 3,
        "Oficina": 1,
        "Taller": 4,
        "Laboratorio": 4
    }
    
    base = base_risk.get(area, 2)  # Valor por defecto
    return min(10, base + num_alertas * 1.5)  # Cada alerta suma 1.5 al nivel de riesgo, máximo 10

# --- LOGIN ---
def login():
    st.sidebar.title("🔐 Iniciar sesión")
    username = st.sidebar.text_input("Usuario")
    password = st.sidebar.text_input("Contraseña", type="password")

    if st.sidebar.button("Ingresar"):
        if username in users and users[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["user"] = username
            st.success(f"Bienvenido, {username}")
        else:
            st.error("Credenciales incorrectas")

# --- BOTÓN DE PAGO ---
def show_payment_button():
    st.write("💳 Esta sección es Premium. Realiza el pago para continuar.")
    if st.button("Pagar con Stripe"):
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {'name': 'Acceso Premium - App detección de riesgos'},
                    'unit_amount': 3000,  # 30 USD
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url='https://TU-APP.streamlit.app?paid=true',
            cancel_url='https://TU-APP.streamlit.app',
        )
        st.markdown(f"[Haz clic aquí para pagar]({session.url})", unsafe_allow_html=True)

# --- PDF ---
def generar_pdf(interpretacion, imagen_path=None, alertas=None):
    pdf = FPDF()
    pdf.add_page()
    
    # Título
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(0, 10, "INFORME DE RIESGOS LABORALES", ln=True, align='C')
    pdf.ln(5)
    
    # Fecha
    pdf.set_font("Arial", size=10)
    fecha_actual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    pdf.cell(0, 5, f"Generado: {fecha_actual}", ln=True)
    pdf.ln(5)
    
    # Contenido principal
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, interpretacion)
    
    # Añadir alertas si existen
    if alertas and len(alertas) > 0:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(0, 10, "ALERTAS DETECTADAS:", ln=True)
        pdf.set_font("Arial", size=12)
        for alerta in alertas:
            pdf.cell(0, 10, f"• {alerta}", ln=True)
    
    # Añadir imagen si existe
    if imagen_path:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(0, 10, "IMAGEN ANALIZADA:", ln=True)
        pdf.image(imagen_path, x=10, y=None, w=180)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# --- INTERPRETACIÓN SST ---
def interpretar_grafico(df):
    area_mayor_riesgo = df.loc[df["Nivel de Riesgo"].idxmax(), "Área"]
    riesgo_max = df["Nivel de Riesgo"].max()
    riesgo_promedio = df["Nivel de Riesgo"].mean()

    texto = f"""📊 INTERPRETACIÓN:
- Área con mayor riesgo: {area_mayor_riesgo} (nivel {riesgo_max})
- Riesgo promedio: {riesgo_promedio:.2f}\n"""

    if riesgo_max >= 8:
        texto += f"""\n⚠️ Riesgo ALTO:
- Intervención inmediata
- Inspecciones
- Verificar EPP
- Reentrenar personal
- Controles de ingeniería"""
    elif 5 <= riesgo_max < 8:
        texto += f"""\n🔶 Riesgo MODERADO:
- Plan preventivo
- Procedimientos seguros
- Señalización
- Monitoreo continuo"""
    else:
        texto += f"""\n🟢 Riesgo BAJO:
- Mantener controles
- Buenas prácticas
- Seguimiento periódico"""

    st.markdown(texto)
    return texto

# --- SISTEMA DE ALERTAS ---
def mostrar_alertas(alertas):
    if not alertas:
        st.success("✅ No se han detectado situaciones de riesgo")
        return
    
    st.error("⚠️ ALERTAS DETECTADAS:")
    for alerta in alertas:
        st.warning(alerta)
    
    if len(alertas) >= 2:
        st.error("🚨 ACCIÓN INMEDIATA REQUERIDA: Múltiples situaciones de riesgo detectadas")

# --- MÓDULO DE DETECCIÓN CON CÁMARA ---
def modulo_deteccion_camara():
    st.subheader("📹 Detección de Riesgos con Cámara")
    st.write("Esta función permite detectar riesgos en tiempo real utilizando la cámara.")
    
    if st.button("Iniciar cámara"):
        st.warning("Para habilitar esta función en producción, se requiere configuración adicional.")
        st.info("En una implementación real, aquí se mostraría el feed de la cámara con detección de riesgos en tiempo real.")

# --- APP PRINCIPAL ---
def main_app():
    st.title("⚠️ Sistema de Detección de Riesgos Laborales")
    
    # Pestañas para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["📊 Análisis de Datos", "🔍 Detección en Imágenes", "📹 Monitoreo en Tiempo Real"])
    
    with tab1:
        st.subheader("📈 Análisis de Datos de Riesgos")
        uploaded_file = st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"], key="excel_uploader")
        
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.subheader("Datos cargados:")
            st.dataframe(df)

            st.subheader("📊 Gráfico de Niveles de Riesgo por Área")
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(df["Área"], df["Nivel de Riesgo"], color="orange")
            
            # Colorear barras según nivel de riesgo
            for i, bar in enumerate(bars):
                riesgo = df["Nivel de Riesgo"].iloc[i]
                if riesgo >= 8:
                    bar.set_color('red')
                elif riesgo >= 5:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
                
            ax.set_xlabel("Área")
            ax.set_ylabel("Nivel de Riesgo")
            ax.set_title("Análisis de Riesgo por Área")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Añadir línea de riesgo promedio
            riesgo_promedio = df["Nivel de Riesgo"].mean()
            ax.axhline(y=riesgo_promedio, color='blue', linestyle='--', label=f'Promedio: {riesgo_promedio:.2f}')
            ax.legend()
            
            st.pyplot(fig)

            interpretacion = interpretar_grafico(df)

            if st.button("📄 Descargar informe en PDF"):
                pdf_path = generar_pdf(interpretacion)
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Descargar PDF", data=f, file_name="informe_riesgos.pdf")
    
    with tab2:
        st.subheader("🔍 Detección de Riesgos en Imágenes")
        st.write("Carga una imagen para detectar posibles situaciones de riesgo.")
        
        uploaded_image = st.file_uploader("Sube una imagen (.jpg, .png, .jpeg)", type=["jpg", "jpeg", "png"], key="image_uploader")
        area_seleccionada = st.selectbox("Selecciona el área", ["Construcción", "Bodega", "Oficina", "Taller", "Laboratorio", "Otra"])
        
        if uploaded_image is not None:
            # Convertir imagen subida a formato numpy
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            imagen = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            
            # Mostrar imagen original
            st.subheader("Imagen Original")
            st.image(imagen_rgb, caption="Imagen cargada", use_column_width=True)
            
            # Procesar imagen
            if st.button("Analizar Riesgos"):
                with st.spinner("Analizando riesgos en la imagen..."):
                    # Detectar riesgos
                    imagen_procesada, alertas = detectar_riesgos(imagen)
                    imagen_procesada_rgb = cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2RGB)
                    
                    # Mostrar resultados
                    st.subheader("Resultados del Análisis")
                    st.image(imagen_procesada_rgb, caption="Detecciones", use_column_width=True)
                    
                    # Mostrar alertas
                    mostrar_alertas(alertas)
                    
                    # Calcular nivel de riesgo del área basado en alertas detectadas
                    nivel_riesgo = calcular_nivel_riesgo(area_seleccionada, len(alertas))
                    
                    # Crear un DataFrame con esta información
                    df_area = pd.DataFrame({
                        "Área": [area_seleccionada],
                        "Nivel de Riesgo": [nivel_riesgo],
                        "Alertas Detectadas": [len(alertas)]
                    })
                    
                    st.subheader("Evaluación de Riesgo")
                    st.dataframe(df_area)
                    
                    # Guardar imagen temporalmente para PDF
                    temp_img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
                    cv2.imwrite(temp_img_path, imagen_procesada)
                    
                    # Generar interpretación
                    interpretacion = f"""📊 INTERPRETACIÓN DEL ÁREA: {area_seleccionada}
- Nivel de riesgo calculado: {nivel_riesgo:.2f}/10
- Número de alertas detectadas: {len(alertas)}
                    
"""
                    
                    if nivel_riesgo >= 8:
                        interpretacion += """⚠️ Riesgo ALTO:
- Intervención inmediata requerida
- Detección de condiciones inseguras
- Verificar equipos de protección personal
- Considerar evacuar el área hasta controlar los riesgos"""
                    elif 5 <= nivel_riesgo < 8:
                        interpretacion += """🔶 Riesgo MODERADO:
- Atención necesaria
- Revisar procedimientos de seguridad
- Verificar sistemas de prevención
- Aumentar supervisión"""
                    else:
                        interpretacion += """🟢 Riesgo BAJO:
- Continuar monitoreando
- Mantener prácticas seguras
- Realizar inspecciones periódicas"""
                    
                    st.markdown(interpretacion)
                    
                    # Botón para descargar informe
                    pdf_path = generar_pdf(interpretacion, temp_img_path, alertas)
                    with open(pdf_path, "rb") as f:
                        st.download_button("⬇️ Descargar Reporte PDF", data=f, file_name=f"informe_{area_seleccionada}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                    
                    # Limpiar archivos temporales
                    os.unlink(temp_img_path)
                    os.unlink(pdf_path)
    
    with tab3:
        modulo_deteccion_camara()

# --- CONTROL DE FLUJO ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    query_params = st.experimental_get_query_params()
    if "paid" in query_params and query_params["paid"][0] == "true":
        main_app()
    else:
        show_payment_button()
