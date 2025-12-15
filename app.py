import heapq
import math
import os
import re
import ydb
import time
import ydb.iam
import logging

from flask_cors import CORS
from datetime import datetime
from flask import Flask, request
from collections import defaultdict
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.DEBUG)

# Создание Flask-приложения
app = Flask(__name__)
CORS(app,
     origins=["http://localhost:63343", "http://localhost:8080",
              "https://web-gis-final.website.yandexcloud.net"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     supports_credentials=False,
     max_age=600)

logging.basicConfig(level=logging.INFO)

# Конфигурация Yandex Database
YDB_ENDPOINT = os.getenv("YDB_ENDPOINT", "grpcs://ydb.serverless.yandexcloud.net:2135")
YDB_DATABASE = os.getenv(
    "YDB_DATABASE", "/ru-central1/b1gi80hd1l574gqup2b4/etnhblgbekdrp1jf6eb6"
)


@app.before_request
def log_everything():
    logging.debug(f"{request.method} {request.path}")
    logging.debug(f"Headers: {dict(request.headers)}")
    if request.method == 'POST':
        try:
            logging.debug(f"Body: {request.get_json()}")
        except:
            logging.debug("Body: Not JSON")


# --- ФУНКЦИИ ДЛЯ РАБОТЫ С ТАБЛИЦАМИ ---
def parse_table_timestamp(table_name, prefix):
    """
    Извлекает timestamp из имени таблицы.
    Формат: {prefix}_YYYYMMDD_HHMMSS
    Возвращает datetime объект или None если формат не соответствует.
    """
    pattern = rf"{prefix}_(\d{{8}})_(\d{{6}})"
    match = re.match(pattern, table_name)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS
        try:
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    return None


def get_latest_table(driver, directory, prefix):
    """
    Находит самую свежую таблицу с заданным префиксом в директории YDB.

    Args:
        driver: YDB драйвер
        directory: путь к директории в YDB
        prefix: префикс таблицы (например, 'nodes' или 'edges')

    Returns:
        Полный путь к таблице или None если таблица не найдена
    """
    try:
        result = driver.scheme_client.list_directory(directory)

        tables_with_timestamps = []
        for table in result.children:
            table_name = table.name
            timestamp = parse_table_timestamp(table_name, prefix)
            if timestamp:
                tables_with_timestamps.append((table_name, timestamp))
                app.logger.info(
                    f"Found table: {table_name} with timestamp: {timestamp}"
                )

        if not tables_with_timestamps:
            app.logger.warning(f"No tables found with prefix '{prefix}' in {directory}")
            return None

        # Сортируем по timestamp и берем самую свежую
        latest_table = sorted(tables_with_timestamps, key=lambda x: x[1], reverse=True)[0]
        full_path = f"{directory}/{latest_table[0]}"

        app.logger.info(
            f"Selected latest table: {latest_table[0]} (timestamp: {latest_table[1]})"
        )
        return full_path

    except Exception as e:
        app.logger.error(f"Error finding latest table with prefix '{prefix}': {e}")
        return None


# --- АЛГОРИТМ ДЛЯ РАСЧЕТА ИЗОХРОН ---
def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Расчет расстояния между двумя точками в метрах с использованием формулы гаверсинуса."""
    R = 6371000

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def dijkstra_multiple_sources(
        graph: Dict[int, List[Tuple[int, float]]],
        node_coords: Dict[int, Tuple[float, float]],
        start_nodes: List[int],
        max_cost: float,
        use_distance: bool = True
) -> Dict[int, float]:
    """
    Алгоритм Дейкстры для нескольких начальных точек.

    Args:
        graph: граф в формате {node_id: [(neighbor_id, cost), ...]}
        node_coords: координаты узлов {node_id: (lon, lat)}
        start_nodes: список начальных узлов
        max_cost: максимальная стоимость пути
        use_distance: если True, используем евклидово расстояние между точками для проверки

    Returns:
        Словарь {node_id: cost} для узлов, достижимых в пределах max_cost
    """
    distances = {}
    priority_queue = []

    for start_node in start_nodes:
        distances[start_node] = 0.0
        heapq.heappush(priority_queue, (0.0, start_node))

    # Основной цикл алгоритма Дейкстры
    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)

        if current_cost > distances.get(current_node, float('inf')):
            continue

        for neighbor, edge_cost in graph.get(current_node, []):
            new_cost = current_cost + edge_cost

            if new_cost < distances.get(neighbor, float('inf')):
                if use_distance:
                    min_start_distance = float('inf')
                    neighbor_lon, neighbor_lat = node_coords.get(neighbor, (0, 0))

                    for start_node in start_nodes:
                        start_lon, start_lat = node_coords.get(start_node, (0, 0))
                        dist = haversine_distance(start_lon, start_lat, neighbor_lon, neighbor_lat)
                        min_start_distance = min(min_start_distance, dist)

                    if new_cost + min_start_distance / 1000.0 > max_cost:
                        continue

                distances[neighbor] = new_cost
                heapq.heappush(priority_queue, (new_cost, neighbor))

    return {node: cost for node, cost in distances.items() if cost <= max_cost}


def create_isochrone_polygon(
        nodes: Dict[int, Tuple[float, float, float]],
        alpha: float = 0.01
) -> List[Tuple[float, float]]:
    """
    Создает полигон изохроны с использованием альфа-формы (упрощенный вариант).

    Args:
        nodes: словарь {node_id: (lon, lat, cost)}
        alpha: параметр альфа-формы (чем больше, тем более выпуклая форма)

    Returns:
        Список точек полигона [(lon, lat), ...]
    """
    if not nodes:
        return []

    points = [(coord[0], coord[1]) for coord in nodes.values()]

    if len(points) <= 2:
        return points

    # Алгоритм Грэхема для выпуклой оболочки
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


    start = min(points, key=lambda p: (p[1], p[0]))
    sorted_points = sorted(points,
                           key=lambda p: (math.atan2(p[1] - start[1], p[0] - start[0]),
                                          (p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2))

    hull = []
    for p in sorted_points:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull


# --- ПОДКЛЮЧЕНИЕ К YDB ---
def get_ydb_driver():
    """Создание драйвера для подключения к YDB."""
    driver_config = ydb.DriverConfig(
        YDB_ENDPOINT,
        YDB_DATABASE,
        credentials=ydb.iam.MetadataUrlCredentials(),
        # credentials=ydb.credentials_from_env_variables(), если запускать локально
        root_certificates=ydb.load_ydb_root_certificate(),
    )
    driver = ydb.Driver(driver_config)
    try:
        driver.wait(fail_fast=True, timeout=5)
        return driver
    except TimeoutError:
        error_msg = "YDB connection failed. Check your endpoint and database path."
        app.logger.error(error_msg)
        raise RuntimeError(error_msg)


# --- РУЧКИ (ENDPOINTS) ---
@app.route("/")
def index():
    return "Hello, world!"


@app.route("/health")
def health():
    try:
        with get_ydb_driver():
            ydb_status = "connected"
    except Exception as e:
        app.logger.error(f"YDB connection error: {str(e)[:100]}")
        ydb_status = "disconnected"

    if ydb_status == "connected":
        return {"status": "healthy", "ydb": ydb_status}, 200
    else:
        return {"status": "unhealthy", "ydb": ydb_status}, 503


@app.post("/calc_isochrones")
def calc_isochrones():
    """
    Основная ручка для выполнения задачи:
    1. Принимает параметры.
    2. Загружает данные из YDB.
    3. Вычисляет изохроны с помощью алгоритма Дейкстры.
    """
    request_id = request.headers.get('X-Request-Id', 'unknown')
    app.logger.info(f"[{request_id}] === START ISOCHRONE REQUEST ===")
    app.logger.info(f"[{request_id}] Headers: {dict(request.headers)}")
    app.logger.info(f"[{request_id}] Body: {request.get_json()}")

    start_total = time.perf_counter()
    data = request.get_json()
    if not data:
        return {"error": "Invalid JSON"}, 400

    points = data.get("points")
    distance = data.get("distance")
    transport_type = data.get("transport_type")
    cost_type = data.get("cost_type")

    if not all(
            [
                isinstance(points, list),
                isinstance(distance, (int, float)),
                isinstance(transport_type, str),
                isinstance(cost_type, str),
            ]
    ):
        return {
            "error": "Missing or invalid parameters: 'points' (list), 'distance' (int), 'transport_type' (str), cost_type (str) are required"
        }, 400

    if cost_type not in ("dist", "time"):
        return {"error": "cost_type must be 'dist' or 'time'"}, 400

    if distance <= 0:
        return {"error": "distance must be positive"}, 400

    app.logger.info(
        f"Received isochrone request: points={points}, distance={distance}, type={transport_type}, cost_type={cost_type}"
    )

    # Определяем путь к таблицам в YDB
    if transport_type == "foot":
        YDB_TABLES_DIR = f"{YDB_DATABASE}/pedestrians/Moscow"
    else:
        YDB_TABLES_DIR = f"{YDB_DATABASE}/highways/Moscow"

    try:
        app.logger.info(f"[{request_id}] Creating YDB driver...")
        ydb_start = time.perf_counter()
        with get_ydb_driver() as ydb_driver:
            # Находим самые свежие таблицы
            ydb_connect_time = time.perf_counter() - ydb_start
            app.logger.info(f"[{request_id}] YDB connected in {ydb_connect_time:.3f}s")

            table_find_start = time.perf_counter()
            ydb_edges_table = get_latest_table(ydb_driver, YDB_TABLES_DIR, "edges")
            ydb_nodes_table = get_latest_table(ydb_driver, YDB_TABLES_DIR, "nodes")
            app.logger.info(f"[{request_id}] Tables found in {time.perf_counter() - table_find_start:.3f}s")

            if not ydb_edges_table or not ydb_nodes_table:
                app.logger.error(f"[{request_id}] Missing tables! Edges: {ydb_edges_table}, Nodes: {ydb_nodes_table}")
                return {"error": "Could not find required tables in YDB"}, 500

            app.logger.info(f"Using edges table: {ydb_edges_table}")
            app.logger.info(f"Using nodes table: {ydb_nodes_table}")

            # Чтение данных из YDB
            with ydb.QuerySessionPool(ydb_driver) as pool:

                def read_edges(session, table_name):
                    """Чтение таблицы edges из YDB."""
                    app.logger.info(f"[{request_id}] Starting edges query from {table_name}")
                    query_start = time.perf_counter()
                    res = []

                    try:
                        query = f"SELECT * FROM `{table_name}`;"
                        app.logger.info(f"[{request_id}] Executing query: {query}")

                        with session.transaction(ydb.QuerySnapshotReadOnly()).execute(
                                query, commit_tx=True
                        ) as result_sets:
                            rows_count = 0
                            for result_set in result_sets:
                                for r in result_set.rows:
                                    cost = float(r.distance) if cost_type == "dist" else float(r.time)
                                    res.append((
                                        int(r.edge_id),
                                        int(r.start_node_id),
                                        int(r.end_node_id),
                                        cost
                                    ))
                                    rows_count += 1

                                    if rows_count % 10000 == 0:
                                        app.logger.info(f"[{request_id}] Loaded {rows_count} edges...")

                        query_time = time.perf_counter() - query_start
                        app.logger.info(f"[{request_id}] Edges query completed: {rows_count} rows in {query_time:.3f}s")

                    except Exception as e:
                        app.logger.error(f"[{request_id}] ERROR in read_edges: {str(e)}", exc_info=True)
                        raise

                    return res

                def read_nodes(session, table_name):
                    """Чтение таблицы nodes из YDB."""
                    res = {}
                    query = f"SELECT * FROM `{table_name}`;"

                    with session.transaction(ydb.QuerySnapshotReadOnly()).execute(
                            query, commit_tx=True
                    ) as result_sets:
                        for result_set in result_sets:
                            for r in result_set.rows:
                                res[int(r.id)] = (float(r.lon), float(r.lat))
                    return res


                edges = pool.retry_operation_sync(
                    callee=read_edges, table_name=ydb_edges_table
                )
                nodes = pool.retry_operation_sync(
                    callee=read_nodes, table_name=ydb_nodes_table
                )

            load_data_time = time.perf_counter() - start_total
            app.logger.info(f"[{request_id}] Data loaded in {load_data_time:.3f}s")
            app.logger.info(f"[{request_id}] Memory usage: {len(edges)} edges, {len(nodes)} nodes")


            if len(edges) == 0:
                app.logger.warning(f"[{request_id}] WARNING: No edges loaded from YDB!")
            if len(nodes) == 0:
                app.logger.warning(f"[{request_id}] WARNING: No nodes loaded from YDB!")

            app.logger.info(
                f"Loaded {len(edges)} edges and {len(nodes)} nodes from YDB"
            )

            # Построение графа
            graph = defaultdict(list)
            for edge_id, start_node, end_node, cost in edges:
                if start_node in nodes and end_node in nodes:
                    graph[start_node].append((end_node, cost))
                    graph[end_node].append((start_node, cost))

            app.logger.info(f"Built graph with {len(graph)} nodes")

            # 2. Поиск ближайших вершин для заданных координат
            start_node_ids = []
            for point in points:
                if isinstance(point, list) and len(point) == 2:
                    try:
                        lon = float(point[0])
                        lat = float(point[1])

                        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                            app.logger.error(
                                f"Invalid coordinates: lon={lon}, lat={lat}"
                            )
                            continue
                    except (ValueError, TypeError):
                        app.logger.error(f"Invalid point values: {point}")
                        continue

                    nearest_node = None
                    min_distance = float('inf')

                    for node_id, (node_lon, node_lat) in nodes.items():
                        dist = haversine_distance(lon, lat, node_lon, node_lat)
                        if dist < min_distance:
                            min_distance = dist
                            nearest_node = node_id

                    if nearest_node is not None:
                        start_node_ids.append(nearest_node)
                        app.logger.info(
                            f"Found nearest node {nearest_node} for point ({lon}, {lat}) "
                            f"distance: {min_distance:.1f}m"
                        )
                    else:
                        app.logger.warning(
                            f"No nearest node found for point ({lon}, {lat})"
                        )
                else:
                    app.logger.error(f"Invalid point format: {point}")

            if not start_node_ids:
                return {"error": "No valid nodes found for given points"}, 400

            app.logger.info(f"Starting nodes: {start_node_ids}")

            app.logger.info(
                f"[{request_id}] Starting Dijkstra with {len(start_node_ids)} sources, graph size: {len(graph)} nodes")

            # 3. Вычисление изохрон с помощью алгоритма Дейкстры
            start_time = time.time()
            reachable_nodes = dijkstra_multiple_sources(
                graph=graph,
                node_coords=nodes,
                start_nodes=start_node_ids,
                max_cost=distance,
                use_distance=(cost_type == "dist")
            )
            elapsed_time = time.time() - start_time

            app.logger.info(
                f"Found {len(reachable_nodes)} reachable nodes in {elapsed_time:.2f} seconds"
            )

            # 4. Формирование результата в формате GeoJSON
            features = []

            # Добавляем точки достижимых узлов
            MAX_RETURN_POINTS = 5000
            points_returned = 0

            for node_id, cost in reachable_nodes.items():
                if points_returned >= MAX_RETURN_POINTS:
                    break

                if node_id in nodes:
                    lon, lat = nodes[node_id]
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat]
                        },
                        "properties": {
                            "node": node_id,
                            "cost": round(cost, 2)
                        }
                    }
                    features.append(feature)
                    points_returned += 1

            # Добавляем полигон изохроны (опционально)
            if reachable_nodes:
                nodes_with_cost = {}
                for node_id, cost in reachable_nodes.items():
                    if node_id in nodes:
                        lon, lat = nodes[node_id]
                        nodes_with_cost[node_id] = (lon, lat, cost)

                polygon_coords = create_isochrone_polygon(nodes_with_cost)

                if len(polygon_coords) >= 3:
                    polygon_coords.append(polygon_coords[0])

                    polygon_feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [polygon_coords]
                        },
                        "properties": {
                            "type": "isochrone",
                            "max_cost": distance,
                            "cost_type": cost_type,
                            "transport_type": transport_type
                        }
                    }
                    features.append(polygon_feature)

            return {
                "type": "FeatureCollection",
                "features": features,
                "metadata": {
                    "reachable_nodes": len(reachable_nodes),
                    "computation_time": round(elapsed_time, 2),
                    "max_cost": distance,
                    "cost_type": cost_type,
                    "start_nodes": start_node_ids
                }
            }

    except Exception as e:
        app.logger.error(f"An error occurred in /calc_isochrones: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}, 500
