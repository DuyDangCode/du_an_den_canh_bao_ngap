import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

// ================= BỘ NÃO MACHINE LEARNING CHẠY TRỰC TIẾP TRÊN TRÌNH DUYỆT (CLIENT ML) =================
// Dùng khi chạy Web Tĩnh (GitHub Pages / Vercel / Netlify) hoặc khi không có máy chủ Python
const DEFAULT_WEIGHTS = {
  bias: -5.0,
  rainfall: 0.25,        // mm/h -> cm ngập
  tide: 0.18,            // cm triều cường -> cm ngập
  drain_level: 0.12,     // % cống đầy -> cm ngập
  drain_speed: 1.20,     // cm/phút tốc độ nước dâng -> cm ngập
  humidity: 0.05,        // % độ ẩm
  pressure: -0.15,       // hPa áp suất giảm -> tăng nguy cơ
  house_elevation: -0.85 // cm cốt nền nhà cao -> giảm ngập
};

class ClientMLEngine {
  constructor() {
    this.storageKey = 'flood_ml_state';
    this.weights = Object.assign({}, DEFAULT_WEIGHTS);
    this.learningRate = 0.01;
    this.trainingEpisodes = 0;
    this.history = [];
    this.loadState();
  }

  loadState() {
    try {
      const saved = localStorage.getItem(this.storageKey);
      if (saved) {
        const data = JSON.parse(saved);
        if (data.weights) this.weights = Object.assign({}, DEFAULT_WEIGHTS, data.weights);
        if (typeof data.trainingEpisodes === 'number') this.trainingEpisodes = data.trainingEpisodes;
        if (Array.isArray(data.history)) this.history = data.history;
      }
    } catch (e) {
      console.warn("Lỗi khi đọc LocalStorage:", e);
    }
  }

  saveState() {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify({
        weights: this.weights,
        trainingEpisodes: this.trainingEpisodes,
        history: this.history.slice(-50)
      }));
    } catch (e) {
      console.warn("Lỗi khi ghi LocalStorage:", e);
    }
  }

  reset() {
    this.weights = Object.assign({}, DEFAULT_WEIGHTS);
    this.trainingEpisodes = 0;
    this.history = [];
    try {
      localStorage.removeItem(this.storageKey);
    } catch (e) {}
  }

  predict(features, thresholds = {}) {
    const th_c1_rain = thresholds.c1_rainfall ?? 40.0;
    const th_c1_tide = thresholds.c1_tide ?? 140.0;
    const th_c2_drain = thresholds.c2_drain ?? 70.0;
    const th_c2_speed = thresholds.c2_drain_speed ?? 2.0;
    const th_c3_floor = thresholds.c3_floor_water ?? 1.0;
    const th_filter_time = thresholds.anti_false_alarm_seconds ?? 30.0;

    const r = Number(features.rainfall) || 0.0;
    const t = Number(features.tide) || 0.0;
    const h = Number(features.humidity) || 70.0;
    const p = (Number(features.pressure) || 1010.0) - 1000.0;
    const dl = Number(features.drain_level) || 0.0;
    const ds = Number(features.drain_speed) || 0.0;
    const elev = Number(features.house_elevation ?? thresholds.house_elevation) || 20.0;
    const fw = Number(features.floor_water) || 0.0;
    const f_dur = Number(features.floor_water_duration) || 0.0;
    const lamp_height = Number(features.lamp_height ?? thresholds.lamp_height) || 80.0;

    const raw_pred = (
      this.weights.bias +
      this.weights.rainfall * r +
      this.weights.tide * (t - 100.0) +
      this.weights.humidity * (h - 60.0) +
      this.weights.pressure * p +
      this.weights.drain_level * dl +
      this.weights.drain_speed * ds +
      this.weights.house_elevation * elev
    );

    const predicted_depth = Math.max(0.0, Math.round(raw_pred * 10) / 10);

    const z = (raw_pred - 2.0) / 5.0;
    const prob = 1.0 / (1.0 + Math.exp(-Math.max(-10.0, Math.min(10.0, z))));
    const probability_pct = Math.round(prob * 1000) / 10;

    const contributions = {
      rainfall_contrib: Math.round(this.weights.rainfall * r * 10) / 10,
      tide_contrib: Math.round(this.weights.tide * Math.max(0.0, t - 100.0) * 10) / 10,
      drain_contrib: Math.round((this.weights.drain_level * dl + this.weights.drain_speed * ds) * 10) / 10,
      elevation_offset: Math.round(this.weights.house_elevation * elev * 10) / 10
    };

    const is_water_on_floor = fw >= th_c3_floor;
    const is_drain_full_or_raining = (dl >= th_c2_drain) || (r >= th_c1_rain);
    const is_duration_passed = f_dur >= th_filter_time;

    let is_false_alarm = false;
    let false_alarm_reason = "";
    let level = 0;
    let level_name = "Ngày bình thường";

    if (is_water_on_floor) {
      if (is_duration_passed && is_drain_full_or_raining) {
        level = 3;
        level_name = "Khẩn cấp (Nước tràn sàn nhà)";
      } else {
        is_false_alarm = true;
        if (!is_duration_passed) {
          false_alarm_reason = `Nước đọng mới ${f_dur.toFixed(0)}s / ngưỡng ${th_filter_time.toFixed(0)}s (đang lau nhà/đổ ly nước)`;
        } else {
          false_alarm_reason = "Có nước trên sàn nhưng cống khô ráo & trời không mưa (nước sinh hoạt)";
        }

        if (dl >= th_c2_drain || ds >= th_c2_speed || predicted_depth > 5.0) {
          level = 2;
          level_name = "Chuẩn bị ứng phó";
        } else if (r >= th_c1_rain || t >= th_c1_tide || probability_pct >= 40.0) {
          level = 1;
          level_name = "Dự báo sớm";
        } else {
          level = 0;
          level_name = "Ngày bình thường";
        }
      }
    } else if (dl >= th_c2_drain || ds >= th_c2_speed || predicted_depth > 5.0) {
      level = 2;
      level_name = "Chuẩn bị ứng phó";
    } else if (r >= th_c1_rain || t >= th_c1_tide || probability_pct >= 40.0) {
      level = 1;
      level_name = "Dự báo sớm";
    } else {
      level = 0;
      level_name = "Ngày bình thường";
    }

    let display_depth = (level === 3 && !is_false_alarm) ? fw : predicted_depth;
    display_depth = Math.max(0.0, display_depth);
    const ruler_pct = Math.min(100.0, Math.round((display_depth / Math.max(10.0, lamp_height)) * 1000) / 10);

    return {
      level,
      level_name,
      predicted_depth_cm: predicted_depth,
      display_depth_cm: display_depth,
      ruler_pct,
      probability_pct,
      contributions,
      is_false_alarm,
      false_alarm_reason,
      zones: {
        manhole: dl >= th_c2_drain ? "Tràn đầy / Nguy cơ cao" : (dl >= 40 ? "Đang dâng" : "Bình thường"),
        living_room: (level === 3 && !is_false_alarm) ? "Bị ngập" : (level >= 2 ? "Mấp mé / Nguy cơ" : "Khô ráo"),
        kitchen_toilet: (level === 3 && !is_false_alarm) ? "Bị ngập" : (level >= 2 ? "Cảnh báo" : "Khô ráo")
      }
    };
  }

  train_online(features, actual_flood_depth_cm, thresholds = {}) {
    const pred_before = this.predict(features, thresholds);
    const y_pred = pred_before.predicted_depth_cm;
    const y_true = Math.max(0.0, Number(actual_flood_depth_cm) || 0.0);

    const error = y_true - y_pred;
    const reward = Math.max(-10.0, 10.0 - Math.abs(error));

    const r = Number(features.rainfall) || 0.0;
    const t = (Number(features.tide) || 0.0) - 100.0;
    const dl = Number(features.drain_level) || 0.0;
    const ds = Number(features.drain_speed) || 0.0;
    const elev = Number(features.house_elevation ?? thresholds.house_elevation) || 20.0;

    const lr = this.learningRate;
    const grad_bias = Math.max(-5.0, Math.min(5.0, error));
    this.weights.bias += lr * grad_bias;
    this.weights.rainfall += lr * 0.01 * error * Math.max(0.0, r);
    this.weights.tide += lr * 0.01 * error * Math.max(0.0, t);
    this.weights.drain_level += lr * 0.01 * error * dl;
    this.weights.drain_speed += lr * 0.05 * error * ds;
    this.weights.house_elevation += lr * 0.02 * (-error) * elev;

    this.weights.rainfall = Math.max(0.01, Math.min(1.0, this.weights.rainfall));
    this.weights.tide = Math.max(0.01, Math.min(1.0, this.weights.tide));
    this.weights.drain_level = Math.max(0.01, Math.min(0.5, this.weights.drain_level));
    this.weights.drain_speed = Math.max(0.1, Math.min(3.0, this.weights.drain_speed));
    this.weights.house_elevation = Math.max(-2.0, Math.min(-0.1, this.weights.house_elevation));

    this.trainingEpisodes += 1;
    const pred_after = this.predict(features, thresholds);

    const record = {
      episode: this.trainingEpisodes,
      actual_cm: y_true,
      pred_before_cm: y_pred,
      pred_after_cm: pred_after.predicted_depth_cm,
      error_before: Math.round(Math.abs(error) * 100) / 100,
      error_after: Math.round(Math.abs(y_true - pred_after.predicted_depth_cm) * 100) / 100,
      reward: Math.round(reward * 100) / 100
    };
    this.history.push(record);
    this.saveState();

    return {
      success: true,
      episodes: this.trainingEpisodes,
      record
    };
  }
}

const clientMLEngine = new ClientMLEngine();
let backendMode = null; // null = đang kiểm tra, true = Backend Python, false = Web Tĩnh (Client ML)

// Trạng thái chung của ứng dụng
const appState = {
  features: {
    rainfall: 5,
    tide: 110,
    humidity: 65,
    pressure: 1012,
    drain_level: 15,
    drain_speed: 0.2,
    floor_water: 0,
    floor_water_duration: 0,
    power_cut: false,
    offline: false,
    torch_detached: false
  },
  thresholds: {
    c1_rainfall: 40.0,
    c1_tide: 140.0,
    c2_drain: 70.0,
    c2_drain_speed: 2.0,
    c3_floor_water: 1.0,
    anti_false_alarm_seconds: 30.0,
    house_elevation: 20.0,
    lamp_height: 80.0
  },
  prediction: {
    level: 0,
    level_name: "Ngày bình thường",
    predicted_depth_cm: 0,
    display_depth_cm: 0,
    ruler_pct: 0,
    probability_pct: 0,
    is_false_alarm: false,
    false_alarm_reason: ""
  },
  muted: false,
  autoRotate: false
};

// 5 Kịch bản mẫu (Presets)
const presets = {
  normal: {
    rainfall: 5, tide: 110, humidity: 65, pressure: 1012,
    drain_level: 15, drain_speed: 0.1, floor_water: 0, floor_water_duration: 0
  },
  early: {
    rainfall: 55, tide: 145, humidity: 85, pressure: 998,
    drain_level: 35, drain_speed: 0.8, floor_water: 0, floor_water_duration: 0
  },
  prep: {
    rainfall: 75, tide: 155, humidity: 90, pressure: 995,
    drain_level: 85, drain_speed: 3.2, floor_water: 0, floor_water_duration: 0
  },
  flood: {
    rainfall: 110, tide: 165, humidity: 95, pressure: 990,
    drain_level: 98, drain_speed: 4.5, floor_water: 8.5, floor_water_duration: 45
  },
  mop: {
    rainfall: 0, tide: 100, humidity: 60, pressure: 1014,
    drain_level: 10, drain_speed: 0.0, floor_water: 2.5, floor_water_duration: 12
  }
};

// ================= THREE.JS SETUP =================
let scene, camera, renderer, controls;
let houseMesh, waterMesh;
let lampGroup, lampLedMesh, lampGlowRing, lampColorLight, lampFloorSpot, torchHeadMesh;

// Bảng 4 màu tương phản cao, tách biệt rõ ràng giữa các cấp độ
const colors = {
  level0: 0x10b981, // Cấp 0 (Bình thường): Xanh lá ngọc an toàn
  level1: 0x0ea5e9, // Cấp 1 (Dự báo sớm): Xanh dương sáng
  level2: 0xf59e0b, // Cấp 2 (Chuẩn bị): Vàng cam tươi
  level3: 0xff0033  // Cấp 3 (Khẩn cấp): Đỏ tươi rực rỡ
};

function init3D() {
  const container = document.getElementById('canvas-container');
  const width = container.clientWidth;
  const height = container.clientHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf1f5f9);

  // Tầm nhìn xa (far = 50000), không có sương mù che khuất, góc nhìn bao quát toàn bộ căn phòng
  camera = new THREE.PerspectiveCamera(50, width / height, 0.5, 50000);
  camera.position.set(0, 78, 88);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 3, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.minDistance = 10;
  controls.maxDistance = 500;
  controls.maxPolarAngle = Math.PI / 2 - 0.01;
  controls.autoRotate = false;

  // Ánh sáng môi trường sáng rõ cho toàn bộ các phòng
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.95);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xfffbf0, 1.3);
  dirLight.position.set(60, 110, 70);
  dirLight.castShadow = true;
  dirLight.shadow.camera.left = -90;
  dirLight.shadow.camera.right = 90;
  dirLight.shadow.camera.top = 90;
  dirLight.shadow.camera.bottom = -90;
  dirLight.shadow.camera.far = 350;
  dirLight.shadow.mapSize.width = 2048;
  dirLight.shadow.mapSize.height = 2048;
  scene.add(dirLight);

  // Mặt đất nền bên dưới đế sàn nhà (y = -1.6)
  const groundGeo = new THREE.PlaneGeometry(260, 260);
  const groundMat = new THREE.MeshStandardMaterial({
    color: 0xe2e8f0,
    roughness: 0.7,
    metalness: 0.1
  });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -1.6;
  ground.receiveShadow = true;
  scene.add(ground);

  // Lưới tọa độ thẩm mỹ
  const gridHelper = new THREE.GridHelper(260, 40, 0xcbd5e1, 0xdbe4ee);
  gridHelper.position.y = -1.55;
  scene.add(gridHelper);

  // Tải mô hình 3D house.stl
  loadHouseSTL();

  // Dựng mô hình 3D Chiếc Đèn Nội Thất
  buildSmartLamp();

  // Dựng mặt nước 3D
  buildWaterPlane();

  window.addEventListener('resize', onWindowResize);
  animate();
}

function loadHouseSTL() {
  const loader = new STLLoader();
  const potentialPaths = [
    './resources/house.stl',
    'resources/house.stl',
    '/resources/house.stl',
    '../resources/house.stl'
  ];

  let pathIdx = 0;
  function tryLoad() {
    if (pathIdx >= potentialPaths.length) {
      console.warn("⚠️ Không tải được house.stl từ mọi đường dẫn, fallback dựng khối nhà tượng trưng");
      const fallbackGeo = new THREE.BoxGeometry(20, 10, 20);
      const fallbackMat = new THREE.MeshStandardMaterial({ color: 0x475569 });
      houseMesh = new THREE.Mesh(fallbackGeo, fallbackMat);
      houseMesh.position.set(0, 5, 0);
      scene.add(houseMesh);
      return;
    }

    const currentPath = potentialPaths[pathIdx++];
    loader.load(
      currentPath,
      (geometry) => {
        geometry.computeVertexNormals();
        geometry.center();

        const scaleFactor = 0.05;
        const material = new THREE.MeshStandardMaterial({
          color: 0xdbe4ee,
          roughness: 0.5,
          metalness: 0.1,
          side: THREE.DoubleSide
        });

        houseMesh = new THREE.Mesh(geometry, material);
        houseMesh.scale.set(scaleFactor, scaleFactor, scaleFactor);
        houseMesh.rotation.x = -Math.PI / 2;
        houseMesh.position.set(0, 6.25, 0);
        houseMesh.castShadow = true;
        houseMesh.receiveShadow = true;

        const edgesGeo = new THREE.EdgesGeometry(geometry, 25);
        const edgesMat = new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.65 });
        const edgesMesh = new THREE.LineSegments(edgesGeo, edgesMat);
        houseMesh.add(edgesMesh);

        scene.add(houseMesh);
        console.log(`✅ Tải house.stl thành công từ: ${currentPath}`);
      },
      undefined,
      () => {
        tryLoad();
      }
    );
  }

  tryLoad();
}

function buildSmartLamp() {
  lampGroup = new THREE.Group();
  lampGroup.position.set(-10, 0, 2); // Đặt ngay giữa không gian phòng khách chính (trên mặt sàn y = 0)

  // Vòng hào quang phát sáng dưới chân đèn để nhìn rõ màu từ xa
  const ringGeo = new THREE.RingGeometry(1.9, 4.2, 32);
  const ringMat = new THREE.MeshBasicMaterial({
    color: colors.level0,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.45
  });
  lampGlowRing = new THREE.Mesh(ringGeo, ringMat);
  lampGlowRing.rotation.x = -Math.PI / 2;
  lampGlowRing.position.y = 0.08;
  lampGroup.add(lampGlowRing);

  // Đế đèn gỗ cao cấp
  const baseGeo = new THREE.CylinderGeometry(1.8, 2.0, 0.5, 24);
  const baseMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.5 });
  const base = new THREE.Mesh(baseGeo, baseMat);
  base.position.y = 0.25;
  base.castShadow = true;
  lampGroup.add(base);

  // Thân đèn LED màu sắc nét, không bị pha nhạt bởi nền phòng
  const lampHeight3D = 9.0;
  const bodyGeo = new THREE.CylinderGeometry(0.85, 0.85, lampHeight3D, 28);
  const bodyMat = new THREE.MeshStandardMaterial({
    color: colors.level0,
    emissive: colors.level0,
    emissiveIntensity: 1.35,
    roughness: 0.15,
    metalness: 0.05
  });
  lampLedMesh = new THREE.Mesh(bodyGeo, bodyMat);
  lampLedMesh.position.y = 0.5 + lampHeight3D / 2;
  lampLedMesh.castShadow = true;
  lampGroup.add(lampLedMesh);

  // Nguồn sáng tỏa màu xung quanh thân đèn
  lampColorLight = new THREE.PointLight(colors.level0, 2.5, 28, 1.5);
  lampColorLight.position.set(0, 5.0, 0);
  lampGroup.add(lampColorLight);

  // Đèn trắng rọi sàn (Floor Downlight)
  lampFloorSpot = new THREE.SpotLight(0xffffff, 0, 22, Math.PI / 2.5, 0.5, 1);
  lampFloorSpot.position.set(0, 1.2, 0);
  lampFloorSpot.target.position.set(0, 0, 0);
  lampGroup.add(lampFloorSpot);
  lampGroup.add(lampFloorSpot.target);

  // Đầu đèn tháo rời bằng nam châm (Torch head)
  const torchGeo = new THREE.CylinderGeometry(0.95, 0.85, 1.2, 24);
  const torchMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.8, roughness: 0.3 });
  torchHeadMesh = new THREE.Mesh(torchGeo, torchMat);
  torchHeadMesh.position.y = 0.5 + lampHeight3D + 0.6;
  torchHeadMesh.castShadow = true;
  lampGroup.add(torchHeadMesh);

  scene.add(lampGroup);
}

function buildWaterPlane() {
  const waterGeo = new THREE.PlaneGeometry(64, 80, 32, 32);
  const waterMat = new THREE.MeshPhysicalMaterial({
    color: 0x0284c7,
    roughness: 0.1,
    metalness: 0.1,
    transmission: 0.85,
    transparent: true,
    opacity: 0.65
  });
  waterMesh = new THREE.Mesh(waterGeo, waterMat);
  waterMesh.rotation.x = -Math.PI / 2;
  waterMesh.position.y = 0.05;
  waterMesh.visible = false;
  scene.add(waterMesh);
}

function update3DLampAndWater(prediction) {
  if (!lampLedMesh) return;

  const level = prediction.level;
  let targetColor = colors.level0;
  if (level === 1) targetColor = colors.level1;
  else if (level === 2) targetColor = colors.level2;
  else if (level === 3) targetColor = colors.level3;

  lampLedMesh.material.color.setHex(targetColor);
  lampLedMesh.material.emissive.setHex(targetColor);
  if (lampGlowRing) lampGlowRing.material.color.setHex(targetColor);
  if (lampColorLight) lampColorLight.color.setHex(targetColor);

  // Đèn rọi sàn chỉ bật khi Khẩn cấp (Cấp 3) và không phải báo giả
  if (level === 3 && !prediction.is_false_alarm) {
    lampFloorSpot.intensity = 4.5;
  } else {
    lampFloorSpot.intensity = 0.0;
  }

  // Đầu đèn tháo rời (Torch)
  if (appState.features.torch_detached) {
    torchHeadMesh.position.y = 0.5 + 9.0 + 3.0; // Nhấc bay lên khỏi thân đèn
    torchHeadMesh.rotation.z = Math.PI / 6;
  } else {
    torchHeadMesh.position.y = 0.5 + 9.0 + 0.6;
    torchHeadMesh.rotation.z = 0;
  }

  // Mặt nước 3D dâng theo mức ngập trên mặt sàn phòng (y = 0)
  const depth = prediction.display_depth_cm;
  if (depth > 0.5) {
    waterMesh.visible = true;
    waterMesh.position.y = Math.min(8.0, 0.08 + depth * 0.18);
  } else {
    waterMesh.visible = false;
  }
}

function onWindowResize() {
  const container = document.getElementById('canvas-container');
  if (!container) return;
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  // Hiệu ứng chớp đỏ rõ nét khi Cấp 3
  if (appState.prediction.level === 3 && lampLedMesh && !appState.prediction.is_false_alarm) {
    const time = Date.now() * 0.008;
    const pulse = (Math.sin(time) + 1) * 0.5;
    lampLedMesh.material.emissiveIntensity = 0.7 + pulse * 1.6;
    if (lampGlowRing) lampGlowRing.material.opacity = 0.25 + pulse * 0.55;
  } else if (lampLedMesh) {
    lampLedMesh.material.emissiveIntensity = 1.35;
    if (lampGlowRing) lampGlowRing.material.opacity = 0.45;
  }

  renderer.render(scene, camera);
}

// ================= ĐỒNG BỘ DỮ LIỆU & API (CHẾ ĐỘ KÉP: BACKEND PYTHON & CLIENT ML) =================

function updateModeBadge(isBackend) {
  const badge = document.getElementById('mode-badge');
  if (!badge) return;
  if (isBackend) {
    badge.className = 'mode-badge backend';
    badge.innerHTML = '🟢 Máy chủ Python';
    badge.title = 'Đang đồng bộ dữ liệu với Backend Python (app/server.py)';
  } else {
    badge.className = 'mode-badge client';
    badge.innerHTML = '🌐 Web Tĩnh (Client ML)';
    badge.title = 'Đang chạy độc lập trên trình duyệt, lưu tiến độ học vào LocalStorage';
  }
}

async function fetchPrediction() {
  if (backendMode !== false) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1200);

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: appState.features,
          thresholds: appState.thresholds
        }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (res.ok) {
        backendMode = true;
        updateModeBadge(true);
        const data = await res.json();
        appState.prediction = data;
        updateUI();
        update3DLampAndWater(data);
        return;
      }
    } catch (err) {
      // Không kết nối được Backend -> Chuyển sang Web Tĩnh Client ML
      backendMode = false;
    }
  }

  // Chế độ Web tĩnh: Chạy bộ não ML trực tiếp trên trình duyệt JavaScript
  backendMode = false;
  updateModeBadge(false);
  const data = clientMLEngine.predict(appState.features, appState.thresholds);
  appState.prediction = data;
  updateUI();
  update3DLampAndWater(data);
}

async function trainOnline(actualCm) {
  if (backendMode) {
    try {
      const res = await fetch('/api/train_online', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: appState.features,
          actual_cm: actualCm
        })
      });
      if (res.ok) {
        const data = await res.json();
        showTrainFeedback(data);
        fetchPrediction();
        return;
      }
    } catch (err) {
      console.warn("Backend lỗi, chuyển sang train tại Client:", err);
      backendMode = false;
      updateModeBadge(false);
    }
  }

  // Cập nhật học tăng cường trực tiếp trên Client
  const data = clientMLEngine.train_online(appState.features, actualCm, appState.thresholds);
  showTrainFeedback(data);
  fetchPrediction();
}

function showTrainFeedback(data) {
  if (data && data.success) {
    const fbMsg = document.getElementById('learn-feedback-msg');
    if (!fbMsg) return;
    fbMsg.style.display = 'block';
    const modeText = backendMode ? "Backend Python" : "Web Tĩnh (LocalStorage)";
    fbMsg.innerHTML = `✅ <strong>Đã cập nhật trọng số (${modeText})!</strong> Cơn mưa thứ ${data.episodes}. Mức thực tế: ${data.record.actual_cm}cm. Dự báo trước: ${data.record.pred_before_cm}cm ➔ Sau khi học: ${data.record.pred_after_cm}cm (Độ chính xác tăng, Sai số giảm còn ${data.record.error_after}cm).`;
  }
}

function resetModel() {
  if (confirm("Bạn có muốn đặt lại toàn bộ trọng số và lịch sử học của mô hình về mặc định ban đầu không?")) {
    clientMLEngine.reset();
    const fbMsg = document.getElementById('learn-feedback-msg');
    if (fbMsg) {
      fbMsg.style.display = 'block';
      fbMsg.innerHTML = '🔄 <strong>Đã đặt lại trọng số mô hình về mặc định ban đầu.</strong>';
    }
    fetchPrediction();
  }
}

async function updateThresholdsAPI() {
  if (backendMode) {
    try {
      await fetch('/api/update_thresholds', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thresholds: appState.thresholds })
      });
    } catch (err) {}
  }
  fetchPrediction();
}

// ================= UI BINDING & EVENTS =================

function updateUI() {
  const p = appState.prediction;

  // Cập nhật Badge & Giá trị ngập
  const badge = document.getElementById('status-badge');
  badge.className = `status-badge badge-level-${p.level}`;
  badge.innerText = `Cấp ${p.level}: ${p.level_name}`;

  document.getElementById('display-depth-val').innerText = `${p.display_depth_cm.toFixed(1)} cm`;
  document.getElementById('display-depth-desc').innerText =
    p.level === 0 ? "Mặt sàn khô ráo • Đèn xanh lá an toàn" :
    p.level === 1 ? "Mưa lớn & Triều cường dâng • Đèn xanh dương (Cảnh báo sớm 60p)" :
    p.level === 2 ? `Cống dâng nhanh • Đèn vàng cam ${p.display_depth_cm.toFixed(1)}cm (${p.ruler_pct}%)` :
    "NƯỚC TRÀN SÀN NHÀ • Đèn chớp đỏ tươi & Bật đèn rọi sàn";

  // Cảnh báo giả lập lau nhà
  const falseBox = document.getElementById('false-alarm-box');
  if (p.is_false_alarm) {
    falseBox.style.display = 'block';
    document.getElementById('false-alarm-text').innerText = p.false_alarm_reason;
  } else {
    falseBox.style.display = 'none';
  }

  // Loa Visual-Only
  const waveform = document.getElementById('speaker-waveform');
  const spkTitle = document.getElementById('speaker-title');
  const spkMsg = document.getElementById('speaker-msg');

  if (appState.muted) {
    waveform.classList.remove('audio-active');
    spkTitle.innerText = "Loa: Đã Tắt Tiếng (Muted)";
    spkMsg.innerText = "Chủ nhà đã thức dậy dọn đồ, tắt còi hú";
  } else if (p.level === 0) {
    waveform.classList.remove('audio-active');
    spkTitle.innerText = "Loa: Im lặng";
    spkMsg.innerText = "Chế độ đèn trang trí nội thất";
  } else if (p.level === 1) {
    waveform.classList.add('audio-active');
    spkTitle.innerText = "Loa: Chuông gió êm dịu";
    spkMsg.innerText = "Báo hiệu chiều tối nay có nguy cơ mưa giông lớn";
  } else if (p.level === 2) {
    waveform.classList.add('audio-active');
    spkTitle.innerText = "Loa: Giọng nói nhắc nhở";
    spkMsg.innerText = "Nhắc lắp ván chắn cửa, kê cao đồ đạc";
  } else if (p.level === 3 && !p.is_false_alarm) {
    waveform.classList.add('audio-active');
    spkTitle.innerText = "Loa: CÒI HÚ KHẨN CẤP!";
    spkMsg.innerText = "Nước đã tràn vào nhà lúc nửa đêm!";
  }

  // Header icon mini
  const hdrLamp = document.getElementById('header-lamp-indicator');
  const hex = p.level === 0 ? "#10b981" : (p.level === 1 ? "#0ea5e9" : (p.level === 2 ? "#f59e0b" : "#ff0033"));
  hdrLamp.style.background = `linear-gradient(to top, ${hex}, transparent)`;
  hdrLamp.style.borderColor = hex;

  // Cập nhật Trang 2
  document.getElementById('ml-level-card').innerText = `Cấp ${p.level}`;
  document.getElementById('ml-level-desc').innerText = p.level_name;
  document.getElementById('ml-prob-card').innerText = `${p.probability_pct}%`;
  document.getElementById('ml-depth-card').innerText = `${p.predicted_depth_cm.toFixed(1)} cm`;
  document.getElementById('ml-ruler-card').innerText = `Cột sáng đèn: ${p.ruler_pct}% thân đèn`;

  if (p.contributions) {
    document.getElementById('contrib-rain').innerText = `+${p.contributions.rainfall_contrib.toFixed(1)} cm`;
    document.getElementById('bar-rain').style.width = `${Math.min(100, p.contributions.rainfall_contrib * 4)}%`;

    document.getElementById('contrib-tide').innerText = `+${p.contributions.tide_contrib.toFixed(1)} cm`;
    document.getElementById('bar-tide').style.width = `${Math.min(100, p.contributions.tide_contrib * 4)}%`;

    document.getElementById('contrib-drain').innerText = `+${p.contributions.drain_contrib.toFixed(1)} cm`;
    document.getElementById('bar-drain').style.width = `${Math.min(100, p.contributions.drain_contrib * 4)}%`;

    document.getElementById('contrib-elev').innerText = `${p.contributions.elevation_offset.toFixed(1)} cm`;
    document.getElementById('bar-elev').style.width = `${Math.min(100, Math.abs(p.contributions.elevation_offset) * 4)}%`;
  }
}

function bindEvents() {
  // Chuyển Tab
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      const targetId = tab.dataset.tab;
      document.getElementById(targetId).classList.add('active');
      if (targetId === 'demo-page') onWindowResize();
    });
  });

  // Tắt còi (Mute)
  const btnMute = document.getElementById('btn-mute');
  btnMute.addEventListener('click', () => {
    appState.muted = !appState.muted;
    btnMute.classList.toggle('muted', appState.muted);
    btnMute.innerText = appState.muted ? "Bật còi lại" : "Tắt còi (Mute)";
    updateUI();
  });

  // Kịch bản mẫu 1-Click
  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const pKey = btn.dataset.preset;
      const data = presets[pKey];
      if (data) {
        Object.assign(appState.features, data);
        syncSlidersToState();
        fetchPrediction();
      }
    });
  });

  // Các thanh trượt Thời tiết & Cảm biến
  bindSlider('input-rainfall', 'val-rainfall', 'rainfall', 'mm/h');
  bindSlider('input-tide', 'val-tide', 'tide', 'cm');
  bindSlider('input-humidity', 'val-humidity', 'humidity', '%');
  bindSlider('input-pressure', 'val-pressure', 'pressure', 'hPa');
  bindSlider('input-drain-level', 'val-drain-level', 'drain_level', '%');
  bindSlider('input-drain-speed', 'val-drain-speed', 'drain_speed', 'cm/phút');
  bindSlider('input-floor-water', 'val-floor-water', 'floor_water', 'cm');
  bindSlider('input-floor-dur', 'val-floor-dur', 'floor_water_duration', 'giây');

  // Nút gạt Thiết bị
  document.getElementById('toggle-power-cut').addEventListener('change', (e) => {
    appState.features.power_cut = e.target.checked;
  });
  document.getElementById('toggle-offline').addEventListener('change', (e) => {
    appState.features.offline = e.target.checked;
  });
  document.getElementById('toggle-torch').addEventListener('change', (e) => {
    appState.features.torch_detached = e.target.checked;
    update3DLampAndWater(appState.prediction);
  });

  // Nút Dạy AI Học Thích Nghi
  document.getElementById('btn-train-online').addEventListener('click', () => {
    const val = parseFloat(document.getElementById('input-actual-depth').value);
    if (!isNaN(val) && val >= 0) {
      trainOnline(val);
    } else {
      alert("Vui lòng nhập số cm ngập thực tế hợp lệ!");
    }
  });

  // Nút Đặt Lại Trọng Số Ban Đầu
  const btnReset = document.getElementById('btn-reset-ml');
  if (btnReset) {
    btnReset.addEventListener('click', () => {
      resetModel();
    });
  }

  // Cấu hình Ngưỡng (Trang 2)
  bindThresholdSlider('th-house-elevation', 'val-th-elev', 'house_elevation', 'cm');
  bindThresholdSlider('th-lamp-height', 'val-th-lamp', 'lamp_height', 'cm');
  bindThresholdSlider('th-c1-rain', 'val-th-rain', 'c1_rainfall', 'mm/h');
  bindThresholdSlider('th-c1-tide', 'val-th-tide', 'c1_tide', 'cm');
  bindThresholdSlider('th-c2-drain', 'val-th-drain', 'c2_drain', '%');
  bindThresholdSlider('th-c3-floor', 'val-th-floor', 'c3_floor_water', 'cm');
  bindThresholdSlider('th-anti-false-time', 'val-th-filter', 'anti_false_alarm_seconds', 'giây');
}

function bindSlider(inputId, valId, key, unit) {
  const input = document.getElementById(inputId);
  const valSpan = document.getElementById(valId);
  input.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    appState.features[key] = val;
    valSpan.innerText = `${val} ${unit}`;
    fetchPrediction();
  });
}

function bindThresholdSlider(inputId, valId, key, unit) {
  const input = document.getElementById(inputId);
  const valSpan = document.getElementById(valId);
  input.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    appState.thresholds[key] = val;
    valSpan.innerText = `${val} ${unit}`;
    updateThresholdsAPI();
  });
}

function syncSlidersToState() {
  const f = appState.features;
  document.getElementById('input-rainfall').value = f.rainfall;
  document.getElementById('val-rainfall').innerText = `${f.rainfall} mm/h`;

  document.getElementById('input-tide').value = f.tide;
  document.getElementById('val-tide').innerText = `${f.tide} cm`;

  document.getElementById('input-humidity').value = f.humidity;
  document.getElementById('val-humidity').innerText = `${f.humidity} %`;

  document.getElementById('input-pressure').value = f.pressure;
  document.getElementById('val-pressure').innerText = `${f.pressure} hPa`;

  document.getElementById('input-drain-level').value = f.drain_level;
  document.getElementById('val-drain-level').innerText = `${f.drain_level} %`;

  document.getElementById('input-drain-speed').value = f.drain_speed;
  document.getElementById('val-drain-speed').innerText = `${f.drain_speed} cm/phút`;

  document.getElementById('input-floor-water').value = f.floor_water;
  document.getElementById('val-floor-water').innerText = `${f.floor_water} cm`;

  document.getElementById('input-floor-dur').value = f.floor_water_duration;
  document.getElementById('val-floor-dur').innerText = `${f.floor_water_duration} giây`;
}

// Khởi chạy khi tài liệu tải xong
window.addEventListener('DOMContentLoaded', () => {
  init3D();
  bindEvents();
  fetchPrediction();
});
