import pandas as pd
import folium
import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Função haversine para distância em km
def haversine_distance(coord1, coord2):
    R = 6371
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# Cria matriz de distâncias
def build_distance_matrix(coords):
    return [[haversine_distance(c1, c2) for c2 in coords] for c1 in coords]


def solve_tsp(distance_matrix, time_limit=10):
    size = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        return int(distance_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)] * 1000)

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    routing.AddDimension(transit_idx, 0, int(1e9), True, 'Distance')

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit

    sol = routing.SolveWithParameters(params)
    if sol:
        idx = routing.Start(0)
        route = []
        while True:
            route.append(manager.IndexToNode(idx))
            if routing.IsEnd(idx): break
            idx = sol.Value(routing.NextVar(idx))
        return route
    return list(range(size))

# Carrega e limpa dados
df = pd.read_csv('dbcaminhoesTESTE.csv', sep=';', encoding='utf-8')
df.columns = df.columns.str.strip()
for c in df.select_dtypes('object'):
    df[c] = df[c].str.strip()

# Conversões numéricas
df['PESO'] = pd.to_numeric(df['PESO'].str.replace(',', '.'), errors='coerce')
df['FATURAMENTO'] = (
    df['FATURAMENTO']
      .str.replace(r'R\$', '', regex=True)
      .str.replace(r'\.', '', regex=True)
      .str.replace(',', '.', regex=True)
      .str.strip()
)
df['FATURAMENTO'] = pd.to_numeric(df['FATURAMENTO'], errors='coerce')

# Estatísticas para legenda
df_group = df.groupby('MOTORISTA').agg(
    PESO_TOTAL=('PESO','sum'),
    CARGA_UTIL=('CARGA','first'),
    VALOR_TOTAL=('FATURAMENTO','sum')
).reset_index()
df_group['USO_%'] = df_group['PESO_TOTAL'] / df_group['CARGA_UTIL'] * 100
faturamento_total = df_group['VALOR_TOTAL'].sum()

# Cores por caminhão
colors = ['red','blue','green','purple','orange','darkred','darkblue','coral','cadetblue','darkpurple','pink','lightblue','lightgreen','gray','black', 'yello']
truck_colors = {t: colors[i%len(colors)] for i,t in enumerate(df['MOTORISTA'].unique())}

# Mapa base
mapa = folium.Map(location=[df['LATITUDE CASA'].mean(), df['LONGITUDE CASA'].mean()], zoom_start=10)

# Loop por caminhão
for truck, grp in df.groupby('MOTORISTA'):
    rows = grp.to_dict('records')
    depot = (rows[0]['LATITUDE CASA'], rows[0]['LONGITUDE CASA'])

    # Separa manhã e diurno
    manha = [r for r in rows if r.get('TURNO RECEBIMENTO','').strip().upper()=='MANHA']
    diurno = [r for r in rows if r not in manha]

    # TSP para manhã
    ordered_manha = []
    last_loc = depot
    if manha:
        coords_m = [depot] + [(r['LATITUDE'],r['LONGITUDE']) for r in manha]
        dm = build_distance_matrix(coords_m)
        route1 = solve_tsp(dm)
        if route1 and route1[-1]==0:
            route1 = route1[:-1]
        ordered_manha = [manha[i-1] for i in route1 if i>0]
        last_loc = coords_m[route1[-1]]

    # TSP para diurno, mas força primeiro a loja mais próxima de last_loc
    ordered_diurno = []
    if diurno:
        coords_d = [last_loc] + [(r['LATITUDE'],r['LONGITUDE']) for r in diurno]
        dd = build_distance_matrix(coords_d)
        route2 = solve_tsp(dd)
        if route2 and route2[-1]==0:
            route2 = route2[:-1]
        # ciclo sem depósito
        cycle = [i for i in route2 if i>0]
        # encontra índice na matriz coords_d com menor distância a last_loc
        distances = [haversine_distance(last_loc, coords_d[i]) for i in cycle]
        nearest_pos = distances.index(min(distances))
        rotated = cycle[nearest_pos:] + cycle[:nearest_pos]
        ordered_diurno = [diurno[i-1] for i in rotated]

    # Combina manhã e diurno
    ordered = ordered_manha + ordered_diurno

    # Plotagem
    fg = folium.FeatureGroup(name=f'Caminhão: {truck}')
    color = truck_colors[truck]
    # depósito
    folium.Marker(depot, popup='VALEMILK-CD', icon=folium.Icon(color=color, icon='home', prefix='fa')).add_to(fg)

    prev = depot
    for idx, r in enumerate(ordered, start=1):
        loc = (r['LATITUDE'], r['LONGITUDE'])
        folium.PolyLine([prev,loc], color=color, weight=2, opacity=0.8).add_to(fg)
        prev = loc
        turno = r.get('TURNO RECEBIMENTO','').strip().upper()
        emoji = '☀️' if turno=='MANHA' else ('🕒' if turno=='DIURNO' else '⚡' if turno=='DIURNO ALERTA' else'🚚')
        icon_html = (
            f"<div style='width:34px;height:34px;background:{color};border-radius:50%;"
            f"display:flex;flex-direction:column;align-items:center;justify-content:center;color:white;'>"
            f"<span style='font-weight:bold;'>{idx}</span><span>{emoji}</span></div>"
        )
        icon = folium.DivIcon(html=icon_html)
        popup = ( f"<b>Motorista:</b>  {r['MOTORISTA']}<br>"
                    f"<b>Ordem:</b> {idx}<br><b>Cliente:</b> {r['NOME FANTASIA']}<br>"
                    f"<b>Turno:</b> {turno}<br><b>Peso:</b> {r['PESO']}<br>"
                    f"<b>Faturamento:</b> R$ {r['FATURAMENTO']}")
        folium.Marker(loc, popup=popup, tooltip=f"{idx} - {r['NOME FANTASIA']} ({turno})", icon=icon).add_to(fg)

    fg.add_to(mapa)

# Legenda e controle
folium.LayerControl().add_to(mapa)
unique_markers = df['NOME FANTASIA'].nunique()
legend = folium.Element(
    '<div style="position:fixed;bottom:50px;left:50px;width:300px;'
    'background:white;border:2px solid grey;z-index:9999;padding:10px;'
    'box-shadow:2px 2px 5px rgba(0,0,0,0.3)">' +
    f'<b>Clientes totais:</b> {unique_markers}<br>' +
    f'<b>Faturamento total:</b> R$ {faturamento_total:.2f}<br>' +
    f'<b>Turnos:</b> ☀️ = Manhã , 🕒 = Diurno, ⚡ = Recebe até as 16h<br>' +
    f'<b>Atualizado:</b> Saida:24/05/2025 <br><br>' +
    ''.join([
        f"<div style='display:flex;align-items:center;margin-bottom:5px;'>"
        f"<div style='width:15px;height:15px;background:{truck_colors.get(row['MOTORISTA'], 'gray')};"
        f"border-radius:50%;margin-right:8px;'></div>"
        f"<b>{row['MOTORISTA']}</b>: R$ {row['VALOR_TOTAL']:.2f} / Uso: {row['USO_%']:.0f}%"
        f"</div>"
        for _, row in df_group.iterrows()
    ]) +
    '</div>'
    
)
mapa.get_root().html.add_child(legend)

# Salva mapa final
mapa.save('rota_otimizada_prioridade_manha.html')
print('Mapa reprocessado com priorização dinâmica e TSP ajustado.')