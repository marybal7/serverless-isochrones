import os
import requests
import time
import math
import ydb
import json
import logging
import hashlib
from datetime import datetime
from ydb.iam import MetadataUrlCredentials
from geopy.distance import distance as geopy_distance

logger = logging.getLogger("osm_import")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def handler(event, context):
    logger.info("Инициализация YDB драйвера...")
    driver = ydb.Driver(
        ydb.DriverConfig(
            endpoint=os.getenv('YDB_ENDPOINT'),
            database=os.getenv('YDB_DATABASE'),
            credentials=MetadataUrlCredentials(),
        )
    )
    driver.wait(timeout=5)
    pool = ydb.SessionPool(driver)
    ydb.logger.setLevel(logging.DEBUG)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_nodes_pedestrian = f"pedestrians/Moscow/nodes_{timestamp}"
    table_edges_pedestrian = f"pedestrians/Moscow/edges_{timestamp}"

    def generate_edge_id(way_id, node_from, node_to, segment_index):
        unique_string = f"{way_id}_{node_from}_{node_to}_{segment_index}"
        return int(hashlib.md5(unique_string.encode()).hexdigest()[:15], 16)

    def create_tables():
        def create(session):
            logger.info("Создание таблиц в YDB...")
            session.execute_scheme(f"""
                CREATE TABLE `{table_nodes_pedestrian}` (
                    id Uint64,
                    lat Float,
                    lon Float,
                    PRIMARY KEY (id)
                );
            """)
            session.execute_scheme(f"""
                CREATE TABLE `{table_edges_pedestrian}` (
                    edge_id Uint64,
                    start_node_id Uint64,
                    end_node_id Uint64,
                    distance Float,
                    time Float,
                    PRIMARY KEY (edge_id)
                );
            """)

        pool.retry_operation_sync(create)
        logger.info(f"Созданы таблицы:\n - {table_nodes_pedestrian}\n - {table_edges_pedestrian}")

    def get_existing_tables():
        full_path = os.path.join(os.getenv("YDB_DATABASE"), "pedestrians/Moscow")
        try:
            result = driver.scheme_client.list_directory(full_path)
            return result.children
        except ydb.SchemeError as e:
            if "Root not found" in str(e):
                logger.warning(f"YDB: путь {full_path} не найден (возможно, ещё нет таблиц)")
                return []
            raise

    def get_oldest_table(prefix):
        tables = get_existing_tables()
        filtered = [t.name for t in tables if t.name.startswith(prefix)]
        if not filtered or len(filtered) < 3:
            return None
        sorted_tables = sorted(filtered)
        return f"pedestrians/Moscow/{sorted_tables[0]}"

    def drop_table_if_exists(relative_path):
        full_path = os.path.join(os.getenv("YDB_DATABASE"), relative_path)

        def drop(session):
            try:
                session.drop_table(full_path)
                logger.info(f"Удалена таблица {full_path}")
            except Exception as e:
                logger.warning(f"Не удалось удалить таблицу {full_path}: {e}")

        pool.retry_operation_sync(drop)

    create_tables()

    try:
        logger.info("Запрос данных с Overpass API...")
        query = """
        [out:json][timeout:300];
        area["name"="Москва"]["boundary"="administrative"]->.searchArea;

        (
            way["highway"~"footway|pedestrian|path"](area.searchArea);
        );

        out body;
        >;
        out skel qt;
        """
        start = time.perf_counter()
        resp = requests.post("https://maps.mail.ru/osm/tools/overpass/api/interpreter", data={"data": query})
        logger.info(f"Ответ от Overpass: {resp.status_code}")

        if not resp.headers.get('Content-Type', '').startswith('application/json'):
            logger.error("ОШИБКА: Ответ не в JSON-формате")
            raise ValueError("Ответ от Overpass не является JSON")

        data = resp.json()
        logger.info(f"JSON распарсен за {time.perf_counter() - start:.2f} сек")

        all_nodes = {}
        ways_pedestrian = []

        for el in data["elements"]:
            if el["type"] == "node":
                all_nodes[el["id"]] = (el["lat"], el["lon"])
        logger.info(f"Собрано {len(all_nodes)} нод за {time.perf_counter() - start:.2f} сек")

        for el in data["elements"]:
            if el["type"] == "way" and "highway" in el.get("tags", {}):
                node_ids = el.get("nodes", [])
                way_id = el["id"]
                for i in range(len(node_ids) - 1):
                    n1, n2 = node_ids[i], node_ids[i + 1]
                    if n1 in all_nodes and n2 in all_nodes:
                        lat1, lon1 = all_nodes[n1]
                        lat2, lon2 = all_nodes[n2]
                        dist = geopy_distance((lat1, lon1), (lat2, lon2)).meters
                        ways_pedestrian.append({
                            "way_id": way_id,
                            "node_from": n1,
                            "node_to": n2,
                            "distance_m": dist,
                            "avg_speed_kmh": 5.0
                        })

        logger.info(f"Всего пешеходных рёбер: {len(ways_pedestrian)}")

        def insert_node_table(table_name, node_ids):
            batch_size = 100000
            node_ids = list(node_ids)
            total_count = 0

            for i in range(0, len(node_ids), batch_size):
                batch = []
                for node_id in node_ids[i:i + batch_size]:
                    lat, lon = all_nodes[node_id]
                    batch.append({"id": node_id, "lat": lat, "lon": lon})

                def insert_single_batch(session):
                    prepared = session.prepare(f"""
                        DECLARE $rows AS List<Struct<id: Uint64, lat: Float, lon: Float>>;
                        UPSERT INTO `{table_name}` SELECT * FROM AS_TABLE($rows);
                    """)
                    session.transaction().execute(prepared, {"$rows": batch}, commit_tx=True)
                    return len(batch)

                try:
                    inserted = pool.retry_operation_sync(insert_single_batch)
                    total_count += inserted
                    logger.info(
                        f"Вставлен батч {i // batch_size + 1} из {math.ceil(len(node_ids) / batch_size)} узлов в {table_name}, всего: {total_count}")
                except Exception as e:
                    logger.error(f"Ошибка вставки батча {i // batch_size + 1} в {table_name}: {e}")

            logger.info(f"Всего вставлено узлов в {table_name}: {total_count}")

        def insert_edge_table(table_name, edges):
            batch_size = 100000
            total_count = 0

            for i in range(0, len(edges), batch_size):
                batch = []
                for j, edge in enumerate(edges[i:i + batch_size]):
                    distance = edge["distance_m"]
                    speed = edge["avg_speed_kmh"]
                    time_sec = distance / (speed * 1000 / 3600)

                    unique_edge_id = generate_edge_id(edge["way_id"], edge["node_from"], edge["node_to"], j)
                    batch.append({
                        "edge_id": unique_edge_id,
                        "start_node_id": edge["node_from"],
                        "end_node_id": edge["node_to"],
                        "distance": distance,
                        "time": time_sec
                    })

                def insert_single_batch(session):
                    prepared = session.prepare(f"""
                        DECLARE $rows AS List<Struct<edge_id: Uint64, start_node_id: Uint64, end_node_id: Uint64, distance: Float, time: Float>>;
                        UPSERT INTO `{table_name}` SELECT * FROM AS_TABLE($rows);
                    """)
                    session.transaction().execute(prepared, {"$rows": batch}, commit_tx=True)
                    return len(batch)

                try:
                    inserted = pool.retry_operation_sync(insert_single_batch)
                    total_count += inserted
                    logger.info(
                        f"Вставлен батч {i // batch_size + 1} из {math.ceil(len(edges) / batch_size)} рёбер в {table_name}, всего: {total_count}")
                except Exception as e:
                    logger.error(f"Ошибка вставки батча {i // batch_size + 1} в {table_name}: {e}")

            logger.info(f"Всего вставлено рёбер в {table_name}: {total_count}")

        pedestrian_node_ids = set()
        for edge in ways_pedestrian:
            pedestrian_node_ids.update([edge["node_from"], edge["node_to"]])
        logger.info(f"Подготовлено {len(pedestrian_node_ids)} pedestrian узлов для вставки")

        insert_node_table(table_nodes_pedestrian, pedestrian_node_ids)
        insert_edge_table(table_edges_pedestrian, ways_pedestrian)

    except Exception as e:
        logger.error(f"ОШИБКА ВО ВРЕМЯ ИМПОРТА: {e}")
        logger.warning("Удаляем только что созданные таблицы...")
        drop_table_if_exists(table_nodes_pedestrian)
        drop_table_if_exists(table_edges_pedestrian)
        raise

    oldest_nodes_table = get_oldest_table("nodes_")
    if oldest_nodes_table:
        drop_table_if_exists(oldest_nodes_table.replace(os.getenv("YDB_DATABASE") + "/", ""))

    oldest_edges_table = get_oldest_table("edges_")
    if oldest_edges_table:
        drop_table_if_exists(oldest_edges_table.replace(os.getenv("YDB_DATABASE") + "/", ""))

    logger.info("Импорт завершён.")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "pedestrian_nodes": len(pedestrian_node_ids),
            "pedestrian_edges": len(ways_pedestrian),
            "tables": {
                "nodes": table_nodes_pedestrian,
                "edges": table_edges_pedestrian
            },
            "deleted_old_tables": {
                "nodes": oldest_nodes_table,
                "edges": oldest_edges_table
            }
        })
    }
