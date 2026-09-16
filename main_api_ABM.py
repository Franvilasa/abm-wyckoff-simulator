import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="API Simulador ABM Wyckoff - Modelo Original", version="4.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationInput(BaseModel):
    seed: Optional[int] = 67
    alpha: float = 0.005
    ruido: str = "Bajo"
    prop_retail: float = 75.0
    prop_contrarian: float = 20.0
    prop_smart_money: float = 5.0
    size_trend: float = 1.0
    size_contrarian: float = 1.0
    size_noise: float = 0.5
    size_algo: float = 0.5
    factor_inv_sm: int = 50
    logica_sm: str = "original"
    umbral_dist_pct: float = 3.0
    k_flujo: int = 10
    umbral_flujo: float = 100.0
    factor_agresividad_sm: float = 2.5
    k_trend: int = 5
    k_contrarian: int = 20
    theta: float = 1.0
    stop_dist: float = 1.0
    prob_stop_loss: float = 70.0
    duracion_max_posicion: int = 20
    alpha_absorcion: float = 0.0005
    tolerancia_pct: float = 0.0001
    capital_inicial: float = 10

class CandleResponse(BaseModel):
    open: float
    high: float
    low: float
    close: float

class SimulationResponse(BaseModel):
    status: str
    candles: List[CandleResponse]
    inventory_sm: List[float]

def decide_SM_flujo(agente, precio_actual, V, fase, inventario_total, senal_flujo, max_inv_total, umbral_flujo, tolerancia_pct, factor_agresividad_sm):
    orden = 0
    nueva_fase = fase
    if fase == 'acumulacion':
        techo_compra = V * (1 + tolerancia_pct)
        if precio_actual <= techo_compra:
            ratio = 1 - (inventario_total / max_inv_total)
            orden = max(1, round(factor_agresividad_sm * ratio))
        if senal_flujo >= umbral_flujo:
            nueva_fase = 'distribucion'
    elif fase == 'distribucion':
        if inventario_total > 0:
            ratio = inventario_total / max_inv_total
            orden = -max(1, round(factor_agresividad_sm * ratio))
        if inventario_total <= 0:
            nueva_fase = 'acumulacion'
    return orden, nueva_fase

def decide_SM_original(agente, precio_actual, V, fase, inventario_total, precio_min_acum, max_inv_total, umbral_dist_pct):
    orden = 0
    nueva_fase = fase

    if fase == 'acumulacion':
        if precio_actual < V:
            ratio = 1 - (inventario_total / max_inv_total)
            orden = max(1, round(2.5 * ratio))

        distancia_pct = (precio_actual - precio_min_acum) / V * 100
        if distancia_pct >= umbral_dist_pct:
            nueva_fase = 'distribucion'

    elif fase == 'distribucion':
        if inventario_total > 0:
            ratio = inventario_total / max_inv_total
            orden = -max(1, round(2.5 * ratio))
        if inventario_total <= 0:
            nueva_fase = 'acumulacion'

    return orden, nueva_fase

def decide_trend(agente, historial_precios, k_trend):
    if agente['activo']:
        return 0
    if len(historial_precios) < k_trend + 1:
        return 0
    ultimos = historial_precios[-(k_trend + 1):]
    movimientos = [np.sign(ultimos[i] - ultimos[i-1]) for i in range(1, len(ultimos))]
    trend = sum(movimientos)
    if trend > 0:
        return 1
    elif trend < 0:
        return -1
    else:
        return 0

def decide_contrarian(agente, historial_precios, k_contrarian, theta):
    if agente['activo']:
        return 0
    if len(historial_precios) < k_contrarian:
        return 0
    ultimos = historial_precios[-k_contrarian:]
    mu = np.mean(ultimos)
    sigma = np.std(ultimos)
    if sigma == 0:
        return 0
    z = (historial_precios[-1] - mu) / sigma
    if z > theta:
        return -1
    elif z < -theta:
        return 1
    else:
        return 0

def decide_noise(agente):
    return np.random.choice([1, -1]) * 0.5

def decide_algo(agente, historial_precios, k_algo=20, umbral_retiro=0.5):
    if len(historial_precios) < k_algo:
        return 0
    volatilidad = np.std(historial_precios[-k_algo:])
    if volatilidad > umbral_retiro:
        return 0
    return np.random.choice([1, -1]) * 0.5

def _cerrar_posicion_y_actualizar_capital(agente, precio_salida, size):
    pnl = agente['lado'] * size * (precio_salida - agente['precio_entrada'])
    agente['capital'] = agente.get('capital', 0.0) + pnl
    if agente['capital'] <= 0:
        agente['quebrado'] = True

def ejecutar_stops(agentes, precio_actual, stop_dist, sizes):
    ordenes_stop = []
    for agente in agentes:
        if not agente.get('activo', False):
            continue
        if not agente.get('tiene_stop', False):
            continue
        nivel_stop = agente['precio_entrada'] + (agente['lado'] * -stop_dist)
        if agente['lado'] == 1 and precio_actual <= nivel_stop:
            size = sizes.get(agente['tipo'], 1.0)
            ordenes_stop.append(-agente['lado'] * size)  # cerrar largo: vender
            _cerrar_posicion_y_actualizar_capital(agente, precio_actual, size)
            agente['activo'] = False
            agente['lado'] = 0
        elif agente['lado'] == -1 and precio_actual >= nivel_stop:
            size = sizes.get(agente['tipo'], 1.0)
            ordenes_stop.append(-agente['lado'] * size)  # cerrar corto: comprar
            _cerrar_posicion_y_actualizar_capital(agente, precio_actual, size)
            agente['activo'] = False
            agente['lado'] = 0
    return ordenes_stop

def ejecutar_modelo_original(params: SimulationInput, guardar_log: bool = False):
    if params.seed is not None:
        np.random.seed(params.seed)

    log_rows = [] if guardar_log else None

    T = 5000
    P_0 = 100.0
    V = 100.0
    sigma_ruido_map = {"Bajo": 0.03, "Medio": 0.05, "Alto": 0.10}
    sigma_ruido = sigma_ruido_map.get(params.ruido, 0.05)
    alpha_mercado = params.alpha
    alpha_absorcion = params.alpha_absorcion

    N_total = 200
    n_sm = max(1, int(N_total * (params.prop_smart_money / 100.0)))
    n_contrarian = max(1, int(N_total * (params.prop_contrarian / 100.0)))
    n_retail_total = N_total - n_sm - n_contrarian
    prop_trend = 80 / 150
    prop_noise = 40 / 150
    prop_algo = 30 / 150
    n_trend = int(n_retail_total * prop_trend)
    n_noise = int(n_retail_total * prop_noise)
    n_algo = n_retail_total - n_trend - n_noise

    k_trend = params.k_trend
    k_contrarian = params.k_contrarian
    theta = params.theta
    stop_dist = params.stop_dist
    factor_agresividad_sm = params.factor_agresividad_sm
    umbral_no_stop = 1 - (params.prob_stop_loss / 100.0)
    duracion_max_posicion = params.duracion_max_posicion
    max_inv_total = params.factor_inv_sm * n_sm
    k_flujo = params.k_flujo
    umbral_flujo = params.umbral_flujo
    umbral_dist_pct = params.umbral_dist_pct

    tolerancia_pct = params.tolerancia_pct
    capital_inicial = float('inf') if params.logica_sm == "original" else params.capital_inicial

    # Diccionario de tamaños por tipo
    sizes = {
        'TREND': params.size_trend,
        'CONT': params.size_contrarian,
        'NOISE': params.size_noise,
        'ALGO': params.size_algo
    }

    # Crear agentes
    agentes = []
    agente_id = 0

    for _ in range(n_sm):
        agentes.append({
            'id': agente_id,
            'tipo': 'SM',
            'inventario': 0,
            'precio_entrada': 0.0,
            'activo': False,
            'max_inv': 50
        })
        agente_id += 1

    for _ in range(n_trend):
        tiene_stop = np.random.random() > umbral_no_stop
        agentes.append({
            'id': agente_id,
            'tipo': 'TREND',
            'inventario': 0,
            'precio_entrada': 0.0,
            'lado': 0,
            'tiene_stop': tiene_stop,
            'activo': False,
            'ticks_en_posicion': 0,
            'max_ticks': duracion_max_posicion if tiene_stop else 1000,
            'capital': capital_inicial,
            'quebrado': False
        })
        agente_id += 1

    for _ in range(n_contrarian):
        tiene_stop = np.random.random() > umbral_no_stop
        agentes.append({
            'id': agente_id,
            'tipo': 'CONT',
            'inventario': 0,
            'precio_entrada': 0.0,
            'lado': 0,
            'tiene_stop': tiene_stop,
            'activo': False,
            'ticks_en_posicion': 0,
            'max_ticks': duracion_max_posicion if tiene_stop else 1000,
            'capital': capital_inicial,
            'quebrado': False
        })
        agente_id += 1

    for _ in range(n_noise):
        agentes.append({'id': agente_id, 'tipo': 'NOISE'})
        agente_id += 1

    for _ in range(n_algo):
        agentes.append({'id': agente_id, 'tipo': 'ALGO'})
        agente_id += 1

    historial_precios = [P_0]
    historial_inventario_SM = [0]
    historial_flujo_mercado = [0.0]
    historial_senal_flujo = [0.0]
    precio_actual = P_0
    fase_SM = 'acumulacion'
    precio_min_acum = P_0

    for t in range(T):
        ordenes = []

        if fase_SM == 'acumulacion':
            precio_min_acum = min(precio_min_acum, precio_actual)

        for agente in agentes:
            if agente.get('quebrado', False):
                continue

            if agente.get('activo', False) and agente['tipo'] in ['TREND', 'CONT']:
                agente['ticks_en_posicion'] += 1
                if agente['ticks_en_posicion'] >= agente['max_ticks']:
                    size = sizes.get(agente['tipo'], 1.0)
                    ordenes.append({'tipo': 'mercado', 'cantidad': -agente['lado'] * size})
                    _cerrar_posicion_y_actualizar_capital(agente, precio_actual, size)
                    agente['activo'] = False
                    agente['lado'] = 0
                    agente['ticks_en_posicion'] = 0
                    continue

            if agente['tipo'] == 'SM':
                inventario_total_SM = sum(a['inventario'] for a in agentes if a['tipo'] == 'SM')
                if params.logica_sm == "original":
                    decision, fase_SM = decide_SM_original(
                        agente, precio_actual, V, fase_SM,
                        inventario_total_SM, precio_min_acum, max_inv_total,
                        umbral_dist_pct
                    )
                else:
                    senal_flujo_actual = historial_senal_flujo[-1]
                    decision, fase_SM = decide_SM_flujo(
                        agente, precio_actual, V, fase_SM,
                        inventario_total_SM, senal_flujo_actual, max_inv_total, umbral_flujo,
                        tolerancia_pct,
                        factor_agresividad_sm
                    )
                if decision != 0:
                    ordenes.append({'tipo': 'limite', 'cantidad': decision})
                    agente['inventario'] += decision

            elif agente['tipo'] == 'TREND':
                decision = decide_trend(agente, historial_precios, k_trend)
                if decision != 0:
                    size = sizes['TREND']
                    cantidad = decision * size
                    ordenes.append({'tipo': 'mercado', 'cantidad': cantidad})
                    agente['activo'] = True
                    agente['lado'] = decision
                    agente['precio_entrada'] = precio_actual
                    agente['ticks_en_posicion'] = 0

            elif agente['tipo'] == 'CONT':
                decision = decide_contrarian(agente, historial_precios, k_contrarian, theta)
                if decision != 0:
                    size = sizes['CONT']
                    cantidad = decision * size
                    ordenes.append({'tipo': 'mercado', 'cantidad': cantidad})
                    agente['activo'] = True
                    agente['lado'] = decision
                    agente['precio_entrada'] = precio_actual
                    agente['ticks_en_posicion'] = 0

            elif agente['tipo'] == 'NOISE':
                decision = decide_noise(agente)
                size = sizes['NOISE']
                cantidad = decision * size
                ordenes.append({'tipo': 'mercado', 'cantidad': cantidad})

            elif agente['tipo'] == 'ALGO':
                decision = decide_algo(agente, historial_precios)
                if decision != 0:
                    size = sizes['ALGO']
                    cantidad = decision * size
                    ordenes.append({'tipo': 'mercado', 'cantidad': cantidad})

        # Ejecutar stops
        ordenes_stop = ejecutar_stops(agentes, precio_actual, stop_dist, sizes)
        for o in ordenes_stop:
            ordenes.append({'tipo': 'mercado', 'cantidad': o})

        flujo_mercado = sum(o['cantidad'] for o in ordenes if o['tipo'] == 'mercado')
        absorcion = sum(o['cantidad'] for o in ordenes if o['tipo'] == 'limite')
        ruido = np.random.normal(0, sigma_ruido)
        precio_nuevo = precio_actual + alpha_mercado * flujo_mercado + alpha_absorcion * absorcion + ruido
        precio_actual = precio_nuevo

        inventario_SM = sum(a['inventario'] for a in agentes if a['tipo'] == 'SM')
        historial_inventario_SM.append(inventario_SM)
        historial_precios.append(precio_actual)
        historial_flujo_mercado.append(float(flujo_mercado))
        senal_flujo = float(np.sum(historial_flujo_mercado[-k_flujo:]))
        historial_senal_flujo.append(senal_flujo)

        
        if guardar_log:
            trend_vivos = [a for a in agentes if a['tipo'] == 'TREND' and not a.get('quebrado', False)]
            cont_vivos = [a for a in agentes if a['tipo'] == 'CONT' and not a.get('quebrado', False)]
            n_trend_quebrados = sum(1 for a in agentes if a['tipo'] == 'TREND' and a.get('quebrado', False))
            n_cont_quebrados = sum(1 for a in agentes if a['tipo'] == 'CONT' and a.get('quebrado', False))
            log_rows.append({
                'tick': t,
                'precio': precio_actual,
                'fase_SM': fase_SM,
                'inventario_SM': inventario_SM,
                'senal_flujo': senal_flujo,
                'flujo_mercado': flujo_mercado,
                'absorcion': absorcion,
                'capital_total_trend': sum(a.get('capital', 0.0) for a in trend_vivos),
                'capital_total_cont': sum(a.get('capital', 0.0) for a in cont_vivos),
                'n_trend_activos': sum(1 for a in trend_vivos if a.get('activo', False)),
                'n_cont_activos': sum(1 for a in cont_vivos if a.get('activo', False)),
                'n_trend_quebrados': n_trend_quebrados,
                'n_cont_quebrados': n_cont_quebrados,
            })

    N_agrupacion = T // 200
    velas_ohlc = []
    inv_comprimido = []
    for i in range(200):
        inicio = i * N_agrupacion
        fin = inicio + N_agrupacion
        segmento_p = historial_precios[inicio:fin]
        segmento_inv = historial_inventario_SM[inicio:fin]
        velas_ohlc.append(CandleResponse(
            open=float(segmento_p[0]),
            high=float(np.max(segmento_p)),
            low=float(np.min(segmento_p)),
            close=float(segmento_p[-1])
        ))
        inv_comprimido.append(float(segmento_inv[-1]))

    return velas_ohlc, inv_comprimido, log_rows

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/v1/simulate", response_model=SimulationResponse)
def run_simulation(payload: SimulationInput):
    try:
        candles, inventory_sm, _ = ejecutar_modelo_original(payload)
        return SimulationResponse(status="success", candles=candles, inventory_sm=inventory_sm)
    except Exception as e:
        return SimulationResponse(status=f"error: {str(e)}", candles=[], inventory_sm=[])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api_ABM:app", host="127.0.0.1", port=8000, reload=True)
