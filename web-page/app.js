let map;
let points = [];
let markers = [];
let isochronesLayer = null;
let moscowBounds;

const cachesByKey = new Map();


let loadingCount = 0;

function showLoading(text = "Считаем изохроны…", subtitle = "Пожалуйста, подождите") {
    loadingCount++;
    const el = document.getElementById("loadingOverlay");
    if (!el) return;

    const titleEl = el.querySelector(".loading-title");
    const subEl = document.getElementById("loadingSubtitle");

    if (titleEl) titleEl.textContent = text;
    if (subEl) subEl.textContent = subtitle;

    el.classList.remove("hidden");
}

function hideLoading() {
    loadingCount = Math.max(0, loadingCount - 1);
    if (loadingCount !== 0) return;

    const el = document.getElementById("loadingOverlay");
    if (!el) return;
    el.classList.add("hidden");
}

function getCache(key) {
    if (!cachesByKey.has(key)) {
        cachesByKey.set(key, {processed: new Set(), best: new Map()});
    }
    return cachesByKey.get(key);
}

function clearAllCaches() {
    cachesByKey.clear();
}

function mergeIntoBest(bestMap, featureCollection, transport) {
    for (const f of (featureCollection?.features || [])) {
        const node = f?.properties?.node;
        const cost = f?.properties?.cost;
        const geom = f?.geometry;
        if (node == null || cost == null || !geom) continue;

        const prev = bestMap.get(node);
        if (!prev || cost < prev.cost) {
            bestMap.set(node, {cost, geometry: geom, transport});
        }
    }
}

function buildCombinedFeatureCollection() {
    const features = [];
    for (const cache of cachesByKey.values()) {
        for (const [node, v] of cache.best.entries()) {
            features.push({
                type: "Feature",
                geometry: v.geometry,
                properties: {node, cost: v.cost, transport: v.transport}
            });
        }
    }
    return {type: "FeatureCollection", features};
}

// Инициализация карты
function initMap() {
    fetch('https://nominatim.openstreetmap.org/search.php?q=Москва&polygon_geojson=1&format=json')
        .then(response => response.json())
        .then(data => {
            const geojson = data[0].geojson;
            const moscowLayer = L.geoJSON(geojson);
            moscowBounds = moscowLayer.getBounds();

            map = L.map('map', {
                attributionControl: false,
                minZoom: 8,
                maxZoom: 45,
                zoom: 18
            }).setView([55.751244, 37.618423], 12).fitBounds(moscowBounds);

            map.zoomControl.setPosition('topright');

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

            moscowLayer.setStyle({color: 'red', weight: 3, fillOpacity: 0}).addTo(map);

            // Инициализация обработчиков после создания карты
            initEventHandlers();
        })
        .catch(error => {
            console.error('Ошибка при загрузке границ Москвы:', error);
            // Если не удалось загрузить границы, создаем карту с центром в Москве
            map = L.map('map').setView([55.751244, 37.618423], 12);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
            initEventHandlers();
        });
}

// Функция для проверки, находится ли точка в пределах Москвы
function isPointInMoscow(latlng) {
    if (!moscowBounds) return true; // Если границы не загрузились, пропускаем проверку
    return moscowBounds.contains(latlng);
}

// Инициализация обработчиков событий
function initEventHandlers() {
    // Обработчик клика по карте
    map.on('click', function (e) {
        addPoint(e.latlng);
    });

    // Обработчик поиска адреса
    document.getElementById('searchBtn').addEventListener('click', searchAddress);

    document.getElementById('searchBtn2').addEventListener('click', searchLatLon);

    // Обработчики кнопок
    document.getElementById('buildBtn').addEventListener('click', buildIsochrones);
    document.getElementById('resetBtn').addEventListener('click', reset);

    document.getElementById('toggleSidebar').addEventListener('click', () => {
        document.getElementById('container').classList.toggle('sidebar-collapsed');
        setTimeout(() => {
            map.invalidateSize();
        }, 300);
    });
}

function searchLatLon() {
    const latitude = document.getElementById('latlonSearch').value;
    const longitude = document.getElementById('latlonSearch2').value;


    if (latitude && longitude) {
        addPoint(L.latLng(latitude, longitude));
    }

    if (!latitude) {
        alert('Введите широту точки');
        return;
    }

    if (!longitude) {
        alert('Введите долготу точки');
        return;
    }
}

// Функция поиска адреса
function searchAddress() {
    const address = document.getElementById('addressSearch').value.trim();

    if (!address) {
        alert('Введите адрес для поиска');
        return;
    }

    const searchUrl = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(address)}&format=json&polygon_geojson=1&bounded=1&viewbox=37.1,56.0,37.9,55.5`;

    fetch(searchUrl)
        .then(response => response.json())
        .then(data => {
            if (data.length === 0) {
                throw new Error('Адрес не найден');
            }

            const result = data[0];
            const lat = parseFloat(result.lat);
            const lon = parseFloat(result.lon);
            const latlng = L.latLng(lat, lon);

            if (!isPointInMoscow(latlng)) {
                throw new Error('Адрес находится за пределами Москвы');
            }

            addPoint(latlng);
            map.setView(latlng, 16);
        })
        .catch(error => {
            alert('Ошибка поиска: ' + error.message);
        });
}

// Функция добавления точки
function addPoint(latlng) {
    // Проверяем, что точка в пределах Москвы
    if (!isPointInMoscow(latlng)) {
        L.popup()
            .setLatLng(latlng)
            .setContent('Точка находится за пределами Москвы')
            .openOn(map);
        return;
    }

    // Проверяем, нет ли уже точки в этом месте
    const existingPoint = findPointAtLocation(latlng);
    if (existingPoint) {
        removePoint(existingPoint.id);
        return;
    }

    // "замораживаем" транспорт на момент добавления точки
    const currentTransport = document.getElementById('travelMode')?.value || 'car';

    const pointId = Date.now();
    points.push({
        id: pointId,
        lat: latlng.lat,
        lng: latlng.lng,
        transport_type: currentTransport,
    });

    const marker = L.marker(latlng, {draggable: true}).addTo(map);

    marker.on('dragend', function (e) {
        const newLatLng = e.target.getLatLng();
        const p = points.find(p => p.id === pointId);

        if (!isPointInMoscow(newLatLng)) {
            if (p) e.target.setLatLng([p.lat, p.lng]);
            L.popup()
                .setLatLng(newLatLng)
                .setContent('Точка должна находиться в пределах Москвы')
                .openOn(map);
            return;
        }

        // обновим координаты
        if (p) {
            p.lat = newLatLng.lat;
            p.lng = newLatLng.lng;
        }
        updatePointsList();

        // так как точка сместилась — кэши стали неверными
        clearAllCaches();
        if (isochronesLayer) {
            map.removeLayer(isochronesLayer);
            isochronesLayer = null;
        }
    });

    // клик по маркеру удаляет точку
    marker.on('click', function () {
        removePoint(pointId);
    });

    markers.push({id: pointId, marker});
    updatePointsList();
}

// Функция для поиска точки в указанном месте
function findPointAtLocation(latlng) {
    const tolerance = 0.0001; // Допустимая погрешность при сравнении координат
    return points.find(point =>
        Math.abs(point.lat - latlng.lat) < tolerance &&
        Math.abs(point.lng - latlng.lng) < tolerance
    );
}

// Удаление точки
function removePoint(id) {
    // Удаляем маркер с карты
    const markerIndex = markers.findIndex(m => m.id === id);
    if (markerIndex !== -1) {
        map.removeLayer(markers[markerIndex].marker);
        markers.splice(markerIndex, 1);
    }

    // Удаляем точку из массива
    points = points.filter(p => p.id !== id);

    updatePointsList();

    // Кэши некорректны после удаления (min мог быть от удалённой точки)
    clearAllCaches();
    if (isochronesLayer) {
        map.removeLayer(isochronesLayer);
        isochronesLayer = null;
    }
}

// Обновление списка точек в сайдбаре
function updatePointsList() {
    const container = document.getElementById('pointsContainer');
    container.innerHTML = '';

    if (points.length === 0) {
        container.innerHTML = '<p>Нет добавленных точек</p>';
        return;
    }

    points.forEach(point => {
        const pointElement = document.createElement('div');
        pointElement.className = 'point-item';
        pointElement.innerHTML = `
      <h3>Точка #${points.indexOf(point) + 1}</h3>
      <div class="point-coords">
        Широта: ${point.lat.toFixed(6)}, Долгота: ${point.lng.toFixed(6)}
      </div>
      <div class="point-coords">
        Транспорт: <b>${point.transport_type}</b>
      </div>
      <span class="remove-point" data-id="${point.id}">×</span>
    `;
        container.appendChild(pointElement);
    });

    // Добавляем обработчики для кнопок удаления
    document.querySelectorAll('.remove-point').forEach(btn => {
        btn.addEventListener('click', function () {
            removePoint(parseInt(this.getAttribute('data-id'), 10));
        });
    });
}

// Функция построения изохрон
async function buildIsochrones() {
    if (points.length === 0) {
        alert('Добавьте хотя бы одну точку на карту');
        return;
    }

    const rangeType = document.getElementById('rangeType').value;
    const rangeValue = parseInt(document.getElementById('rangeValue').value, 10);

    if (!rangeValue || rangeValue <= 0) {
        alert('Значение диапазона должно быть положительным');
        return;
    }

    const cost_type = (rangeType === 'time') ? 'time' : 'dist';
    const distance = (rangeType === 'time') ? (rangeValue * 60) : rangeValue;

    // группируем точки по транспорту, который записан в точке
    const groups = new Map(); // transport -> [points]
    for (const p of points) {
        const t = p.transport_type || 'car';
        if (!groups.has(t)) groups.set(t, []);
        groups.get(t).push(p);
    }

    const API_BASE = 'https://bbagu22j83dscsoqivur.containers.yandexcloud.net';


    showLoading("Считаем изохроны…", `Точек: ${points.length}. Тип: ${cost_type}.`);

    const buildBtn = document.getElementById('buildBtn');
    const resetBtn = document.getElementById('resetBtn');
    if (buildBtn) buildBtn.disabled = true;
    if (resetBtn) resetBtn.disabled = true;

    try {
        console.log("buildIsochrones grouped transports:", [...groups.keys()]);

        for (const [transport, pts] of groups.entries()) {
            const key = `${transport}|${cost_type}|${distance}`;
            const cache = getCache(key);

            const newPts = pts.filter(p => !cache.processed.has(p.id));
            console.log("key:", key, "newPts:", newPts.length, "processed:", cache.processed.size);

            if (newPts.length === 0) continue;

            const subtitle = document.getElementById("loadingSubtitle");
            if (subtitle) subtitle.textContent = `Новых точек: ${newPts.length}…`;

            const apiBody = {
                transport_type: transport,
                cost_type,
                distance,
                points: newPts.map(p => [p.lng, p.lat]),
            };

            const resp = await fetch(`${API_BASE}/calc_isochrones`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(apiBody)
            });

            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data?.error || data?.message || `HTTP ${resp.status}`);
            }

            mergeIntoBest(cache.best, data, transport);
            newPts.forEach(p => cache.processed.add(p.id));
        }

        displayIsochrones(buildCombinedFeatureCollection());

    } catch (e) {
        console.error(e);
        alert('Ошибка при построении изохрон: ' + e.message);
    } finally {
        hideLoading();
        if (buildBtn) buildBtn.disabled = false;
        if (resetBtn) resetBtn.disabled = false;
    }
}


// Функция отображения изохрон на карте
function displayIsochrones(geojsonData) {
    if (isochronesLayer) map.removeLayer(isochronesLayer);

    isochronesLayer = L.geoJSON(geojsonData, {
        pointToLayer: (feature, latlng) => {
            return L.circleMarker(latlng, {
                radius: 4,
                weight: 1,
                opacity: 1,
                fillOpacity: 0.7
            });
        },
        onEachFeature: (feature, layer) => {
            const cost = feature?.properties?.cost;
            const tr = feature?.properties?.transport;
            layer.bindPopup(`transport: ${tr}<br>cost: ${cost}`);
        }
    }).addTo(map);

    if (isochronesLayer.getBounds().isValid()) {
        map.fitBounds(isochronesLayer.getBounds());
    }
}

// Функция сброса
function reset() {
    // Удаляем все маркеры
    markers.forEach(markerData => {
        map.removeLayer(markerData.marker);
    });

    // Удаляем изохроны
    if (isochronesLayer) {
        map.removeLayer(isochronesLayer);
        isochronesLayer = null;
    }

    // Очищаем массивы
    points = [];
    markers = [];

    // Очищаем кэши
    clearAllCaches();

    // Обновляем список точек
    updatePointsList();
}

document.addEventListener('DOMContentLoaded', function () {
    initMap();
    updatePointsList();
});
