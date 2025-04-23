import pandas as pd
import folium
import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Função para calcular distância haversine entre dois pontos (em km)
def haversine_distance(coord1, coord2):
    R = 6371
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# Constrói matriz de distâncias para uma lista de coordenadas
def build_distance_matrix(coords):
    return [[haversine_distance(c1, c2) for c2 in coords] for c1 in coords]

# Resolve TSP (Circuito) e retorna ordem de índices visitados, começando em 0
def solve_tsp(distance_matrix, time_limit=10):
    size = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)
    routing.AddDimension(transit_index, 0, int(1e9), True, "Distance")

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = time_limit

    solution = routing.SolveWithParameters(search_params)
    if solution:
        index = routing.Start(0)
        route = []
        while True:
            node = manager.IndexToNode(index)
            route.append(node)
            if routing.IsEnd(index):
                break
            index = solution.Value(routing.NextVar(index))
        return route
    return list(range(size))

# -- Leitura e preparação dos dados --
df = pd.read_csv('dbcaminhoes.csv', sep=';', encoding='utf-8')
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=['object']):
    df[col] = df[col].str.strip()

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

# Agrupamento para legenda
df_group = df.groupby('CAMINHAO').agg(
    PESO_TOTAL=('PESO', 'sum'),
    CARGA_UTIL=('CARGA', 'first'),
    VALOR_TOTAL=('FATURAMENTO', 'sum')
).reset_index()
df_group['USO_%'] = df_group['PESO_TOTAL'] / df_group['CARGA_UTIL'] * 100
faturamento_total = df_group['VALOR_TOTAL'].sum()

# Cores por caminhão
colors = ['red','blue','green','purple','orange','darkred','darkblue','darkgreen','cadetblue','darkpurple','pink','lightblue','lightgreen','gray','black','lightgray','coral' ]
truck_colors = {truck: colors[i % len(colors)] for i, truck in enumerate(df['CAMINHAO'].unique())}

# Mapa base
mapa = folium.Map(location=[-3.8666699, -38.5773332], zoom_start=10)

# Plot por caminhão: otimização + marcador com número e ícone de turno e priorização manhã
for truck, group in df.groupby('CAMINHAO'):
    rows = group.to_dict('records')
    depot = (rows[0]['LATITUDE CASA'], rows[0]['LONGITUDE CASA'])
    points = [(r['LATITUDE'], r['LONGITUDE']) for r in rows]

    # Solução TSP
    coords = [depot] + points
    dist_matrix = build_distance_matrix(coords)
    route_idx = solve_tsp(dist_matrix)
    if route_idx and route_idx[-1] == 0:
        route_idx = route_idx[:-1]

    # Ordem inicial pelo TSP
    ordered = [rows[i-1] for i in route_idx if i != 0]
    # Prioriza lojas com turno MANHA
    manha = [r for r in ordered if r.get('TURNO RECEBIMENTO','').strip().upper() == 'MANHA']
    diurno = [r for r in ordered if r.get('TURNO RECEBIMENTO','').strip().upper() != 'MANHA']
    ordered = manha + diurno

    feature_group = folium.FeatureGroup(name=f"Caminhão: {truck}")
    truck_color = truck_colors.get(truck, 'gray')

    # Depósito
    folium.Marker(
        location=depot,
        popup="VALEMILK-CD",
        icon=folium.Icon(color=truck_color, icon='home', prefix='fa')
    ).add_to(feature_group)

    prev = depot
    for idx, r in enumerate(ordered, start=1):
        dest = (r['LATITUDE'], r['LONGITUDE'])
        folium.PolyLine(locations=[prev, dest], color=truck_color, weight=2, opacity=0.8).add_to(feature_group)
        prev = dest

        # Seleção de emoji de turno
        turno = r.get('TURNO RECEBIMENTO', '').strip().upper()
        if turno == 'MANHA':
            emoji = '☀️'
        elif turno == 'DIURNO':
            emoji = '🌤️'
        else:
            emoji = '🚚'

        # DivIcon com número e emoji
        html = (
            f"<div style='width:34px; height:34px; background:{truck_color};"
            f"border-radius:50%; display:flex; align-items:center; justify-content:center;"
            f"flex-direction:column; color:white; font-size:12px;'>"
            f"<span style='font-weight:bold;'>{idx}</span>"
            f"<span>{emoji}</span>"
            f"</div>"
        )
        icon = folium.DivIcon(html=html)

        popup = (
            f"<b>Placa:</b>  {r['CAMINHAO']}<br>"
            f"<b>Ordem:</b> {idx}<br><b>Cliente:</b> {r['NOME FANTASIA']}<br>"
            f"<b>Turno:</b> {turno}<br><b>Peso:</b> {r['PESO']}<br>"
            f"<b>Faturamento:</b> R$ {r['FATURAMENTO']}"
        )
        folium.Marker(
            location=dest,
            popup=popup,
            tooltip=f"{idx} - {r['NOME FANTASIA']} ({turno})",
            icon=icon
        ).add_to(feature_group)

    feature_group.add_to(mapa)

# Legenda e controle
folium.LayerControl().add_to(mapa)
legend = folium.Element(
    f"""
<div style="position:fixed; bottom:50px; left:50px; width:300px; background:white; border:2px solid grey; z-index:9999; padding:10px; box-shadow:2px 2px 5px rgba(0,0,0,0.3)">
    <b>🏬 Clientes totais:</b> {len(df)}<br>
    <b>💰 Faturamento Total:</b> R$ {faturamento_total:.2f}<br>
    <b>🔄 Atualizado:</b> {pd.Timestamp.today().strftime('%d/%m/%Y')}<br><br>
""" + "".join([
        f"<b>{row['CAMINHAO']}</b>: R$ {row['VALOR_TOTAL']:.2f} / Uso: {row['USO_%']:.0f}%<br>"
        for _, row in df_group.iterrows()
    ]) + "</div>"
)
mapa.get_root().html.add_child(legend)

# Salvar mapa otimizado com ordem e prioridade de turno
mapa.save('rota_otimizada_prioridade_manha.html')
print("Mapa otimizado com prioridade para turno manhã salvo como rota_otimizada_prioridade_manha.html")