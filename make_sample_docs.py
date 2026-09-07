"""Genera un corpus de prueba ficticio (dominio fintech) en ./sample_docs.

El contenido es inventado y sirve unicamente para demostrar el pipeline.
"""
from pathlib import Path

DOCS = {
"politica_transferencias.md": """# Politica de Transferencias Electronicas

## Alcance
Esta politica aplica a todas las transferencias electronicas de fondos originadas por clientes
personas fisicas y morales a traves de los canales digitales de la institucion.

## Limites operativos
El limite de transferencia diario para cuentas de personas fisicas con verificacion basica es de
25.000 pesos. Las cuentas con verificacion reforzada tienen un limite diario de 500.000 pesos.
Las cuentas de personas morales operan con un limite diario de 2.000.000 de pesos.
Cualquier operacion que exceda el limite diario debe ser autorizada por el area de riesgo operativo.

## Horarios
Las transferencias interbancarias se procesan en ventanas de liquidacion de 07:00 a 17:30 en dias
habiles. Las operaciones ingresadas fuera de ese horario se liquidan el siguiente dia habil.
Las transferencias entre cuentas de la misma institucion se acreditan de forma inmediata, los 365
dias del ano.

## Reversas
Una transferencia acreditada solo puede reversarse con autorizacion expresa del beneficiario o por
orden de autoridad competente. El plazo maximo para iniciar una solicitud de reversa por error de
digitacion es de 30 dias naturales contados desde la fecha de la operacion.
""",

"onboarding_digital.md": """# Procedimiento de Onboarding Digital

## Objetivo
Estandarizar el alta remota de clientes cumpliendo los requisitos de identificacion no presencial.

## Etapas del proceso
El onboarding digital consta de cinco etapas: captura de datos, validacion documental, prueba de
vida, verificacion contra listas restrictivas y activacion de la cuenta.

## Documentacion requerida
Para personas fisicas se requiere identificacion oficial vigente con fotografia, comprobante de
domicilio con antiguedad no mayor a tres meses y una selfie con prueba de vida activa.
Para personas morales se requiere acta constitutiva, poder del representante legal, identificacion
del representante y comprobante de domicilio fiscal.

## Tiempos de respuesta
La validacion documental automatica debe resolverse en menos de 90 segundos. Los casos derivados a
revision manual tienen un acuerdo de nivel de servicio de 24 horas habiles.
La tasa de aprobacion automatica objetivo es del 82 por ciento de las solicitudes recibidas.

## Rechazos
Una solicitud rechazada por calidad de imagen puede reintentarse hasta tres veces en un periodo de
24 horas. Un rechazo por coincidencia en listas restrictivas es definitivo y se escala al area de
cumplimiento.
""",

"prevencion_fraude.md": """# Manual de Prevencion de Fraude Transaccional

## Modelo de scoring
Cada transaccion recibe un puntaje de riesgo entre 0 y 1000 calculado por un modelo de clasificacion
que se reentrena mensualmente. Las variables de mayor peso son la antiguedad del beneficiario, la
desviacion respecto al monto habitual del cliente, el dispositivo utilizado y la geolocalizacion.

## Umbrales de decision
Las transacciones con puntaje menor a 400 se aprueban automaticamente. Entre 400 y 700 se solicita
un segundo factor de autenticacion. Por encima de 700 la operacion se bloquea y se genera un caso
para el equipo de monitoreo.

## Indicadores de desempeno
El objetivo del modelo es mantener una tasa de falsos positivos por debajo del 2 por ciento sin que
la deteccion de fraude confirmado caiga por debajo del 88 por ciento de los casos.
La perdida por fraude tolerada es de 8 puntos base sobre el volumen transaccionado mensual.

## Gestion de casos
Un caso bloqueado debe ser revisado en un maximo de 4 horas. Si no se contacta al cliente en ese
plazo, la operacion se cancela y se notifica por el canal registrado.
""",

"arquitectura_datos.md": """# Lineamientos de Arquitectura de Datos

## Capas
La plataforma de datos se organiza en tres capas: raw, staging y analytics. La capa raw almacena el
dato tal como llega de la fuente, sin transformaciones, y es inmutable.
La capa staging aplica limpieza, tipado y deduplicacion. La capa analytics expone modelos
dimensionales con esquema estrella para consumo de reporteria.

## Calidad de datos
Todo pipeline debe implementar cuatro controles minimos: validacion de tipos, deteccion de
duplicados, reconciliacion de conteo de filas entre origen y destino, y registro de ejecuciones
fallidas con su traza de error.
Un pipeline que falle la reconciliacion no publica en la capa analytics.

## Retencion
Los datos de la capa raw se conservan por 24 meses. Los modelos de la capa analytics se conservan
por 60 meses por requerimiento regulatorio.

## Informacion sensible
Los datos personales identificables deben enmascararse antes de salir de la capa staging. Ningun
ambiente de desarrollo puede contener datos productivos sin anonimizar.
""",

"soporte_incidencias.md": """# Gestion de Incidencias de Soporte Tecnico

## Clasificacion por severidad
Las incidencias se clasifican en cuatro niveles. Severidad 1 corresponde a la caida total de un
servicio critico. Severidad 2 a la degradacion de un servicio critico. Severidad 3 a fallas en
funcionalidades no criticas. Severidad 4 a consultas y solicitudes de informacion.

## Tiempos de atencion
El tiempo de respuesta comprometido es de 15 minutos para severidad 1, 1 hora para severidad 2,
8 horas habiles para severidad 3 y 48 horas habiles para severidad 4.
El tiempo de resolucion objetivo para severidad 1 es de 4 horas.

## Escalamiento
Una incidencia de severidad 1 que supere las 2 horas sin diagnostico se escala automaticamente al
lider tecnico y al responsable de operaciones.

## Post mortem
Toda incidencia de severidad 1 o 2 requiere un analisis post mortem documentado dentro de los 5 dias
habiles posteriores al cierre, con causa raiz identificada y acciones correctivas asignadas.
""",
}


def main() -> None:
    folder = Path("sample_docs")
    folder.mkdir(exist_ok=True)
    for name, content in DOCS.items():
        (folder / name).write_text(content, encoding="utf-8")
    print(f"{len(DOCS)} documentos escritos en {folder}/")


if __name__ == "__main__":
    main()
