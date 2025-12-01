import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";

// --- GLOBAL VARIABLES ---
let scene, camera, renderer, composer, controls;
let raycaster, pointer, clock;
let clusterParticles = [];
let activeLines = [];
let bloomLayer;

// NEURAL PATHFINDING
let neuralPaths = [];
let hoveredNode = null;
let pathParticles = [];

// QUANTUM MERGE STATE
let quantumMerge = {
  blackHole: null,
  particlePool: [],
  active: false,
};

// State
let linesEnabled = false;

// --- CONFIGURATION ---
const PALETTE = {
  highlight: 0xffd700,
  ghost: 0x8899ac,
  // Vibrant, glowing colors for clusters
  default: [
    0xff00ff, // Bright magenta
    0x00ffff, // Cyan
    0xffff00, // Yellow
    0xff0080, // Hot pink
    0x00ff00, // Lime green
    0xff8000, // Orange
    0x8000ff, // Purple
    0x00ff80, // Spring green
  ],
  neural: 0x00ffff,
};

export function initUniverse(containerId, data, onNodeClick) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.error("❌ Universe container not found:", containerId);
    return null;
  }

  console.log("🚀 Initializing Universe with", data?.length || 0, "points");

  if (renderer) {
    renderer.dispose();
    if (container.contains(renderer.domElement)) {
      container.removeChild(renderer.domElement);
    }
  }

  // Reset
  clusterParticles = [];
  activeLines = [];
  neuralPaths = [];
  pathParticles = [];
  quantumMerge = {
    blackHole: null,
    particlePool: [],
    active: false,
  };
  clock = new THREE.Clock();

  // Scene & Bloom Layer
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x05010a, 0.0015);
  bloomLayer = new THREE.Layers();
  bloomLayer.set(1);

  // Camera
  camera = new THREE.PerspectiveCamera(
    60,
    container.clientWidth / container.clientHeight,
    0.1,
    5000
  );
  camera.position.set(0, 60, 120);

  // Renderer
  renderer = new THREE.WebGLRenderer({
    antialias: false,
    alpha: true,
    powerPreference: "high-performance",
  });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ReinhardToneMapping;
  renderer.toneMappingExposure = 6.0;
  container.appendChild(renderer.domElement);

  console.log(
    "✅ Renderer initialized, canvas size:",
    container.clientWidth,
    "x",
    container.clientHeight
  );

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.5;
  controls.maxDistance = 2000;

  // Bloom with STRONGER intensity for glowing effect
  composer = new EffectComposer(renderer);
  const renderPass = new RenderPass(scene, camera);
  composer.addPass(renderPass);

  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    2.5, // Strength: increased from 1.5 to 2.5
    0.8, // Radius: increased from 0.4 to 0.8
    0.3 // Threshold: lowered from 0.85 to 0.3 (more bloom)
  );
  bloomPass.threshold = 0;
  bloomPass.strength = 2.5;
  bloomPass.radius = 0.8;
  bloomPass.renderToScreen = false;
  composer.addPass(bloomPass);

  // Brighter lighting for vibrant colors
  scene.add(new THREE.AmbientLight(0x606060, 1.5)); // Increased intensity

  const mainLight = new THREE.PointLight(0xffffff, 3, 4000); // Stronger light
  mainLight.position.set(0, 200, 200);
  scene.add(mainLight);

  const fillLight = new THREE.PointLight(0x8888ff, 1.5, 3000);
  fillLight.position.set(-200, -100, -200);
  scene.add(fillLight);

  createStarfield();

  // Data
  if (data && data.length > 0) {
    console.log("🌌 Generating galaxy with", data.length, "points");
    generateGalaxy(data);
  } else {
    console.warn("⚠️ No data provided to Universe Map");
  }

  // Events
  raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 3;
  pointer = new THREE.Vector2();

  window.addEventListener("resize", onWindowResize);
  container.addEventListener("mousemove", onPointerMove);
  container.addEventListener("click", (e) => onMouseClick(e, onNodeClick));

  animate();

  console.log("✅ Universe initialization complete!");

  return {
    setOrbit: (enabled) => {
      if (controls) controls.autoRotate = enabled;
    },
    setLines: (enabled) => {
      linesEnabled = enabled;
      activeLines.forEach((l) => (l.visible = enabled));
      neuralPaths.forEach((p) => (p.visible = enabled));
    },
    performQuantumMerge: performQuantumMerge,
    isQuantumMergeActive: () => quantumMerge.active,
    findSpriteByPath: (path) => {
      const filename = path.split('/').pop();
      return clusterParticles.find((p) => 
        p.userData.path === path || 
        p.userData.path.endsWith(filename) ||
        p.userData.filename === filename
      );
    },
    findSpritesByClusterAndExclude: (cluster, excludePath) => {
      const excludeFilename = excludePath.split('/').pop();
      return clusterParticles.filter(
        (p) => p.userData.cluster === cluster && 
              p.userData.path !== excludePath &&
              !p.userData.path.endsWith(excludeFilename)
      );
    }
  };
}

// --- CORE HELPERS ---
function createStarfield() {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(9000);
  for (let i = 0; i < 9000; i++) positions[i] = (Math.random() - 0.5) * 3000;
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  scene.add(
    new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: 2,
        color: 0x555555,
        transparent: true,
        opacity: 0.6,
      })
    )
  );
  console.log("✨ Starfield created");
}

function generateGalaxy(data) {
  console.log("=== GENERATE GALAXY ===");
  console.log("Data points:", data.length);

  // Create GLOWING sprite texture with brighter core
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");

  // Multi-layer glow effect
  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.1, "rgba(255,255,255,0.9)");
  grad.addColorStop(0.3, "rgba(255,255,255,0.8)");
  grad.addColorStop(0.6, "rgba(255,255,255,0.5)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 64, 64);

  const spriteTexture = new THREE.CanvasTexture(canvas);

  const clusters = {};
  data.forEach((p) => {
    if (!clusters[p.cluster]) clusters[p.cluster] = [];
    clusters[p.cluster].push(p);
  });

  console.log("Clusters found:", Object.keys(clusters).length);
  let totalSprites = 0;
  
  Object.keys(clusters).forEach((key, index) => {
    const isNoise = key === "Noise/Unique";
    const colorHex = isNoise
      ? 0x666666
      : PALETTE.default[index % PALETTE.default.length];

    const material = new THREE.SpriteMaterial({
      map: spriteTexture,
      color: colorHex,
      transparent: true,
      opacity: isNoise ? 0.3 : 1.0, // Full opacity for clusters
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    clusters[key].forEach((item) => {
      const sprite = new THREE.Sprite(material.clone());
      const x = item.x * 2,
        y = item.y * 2,
        z = item.z * 2;
      sprite.position.set(x, y, z);

      // sizes for visibility
      const size = item.is_best ? 40 : isNoise ? 6 : 20;
      sprite.scale.set(size, size, 1);

      sprite.userData = {
        ...item,
        originalPos: new THREE.Vector3(x, y, z),
        originalColor: colorHex,
        originalScale: size,
        isNoise: isNoise,
      };

      sprite.layers.enable(1);
      scene.add(sprite);
      clusterParticles.push(sprite);
      totalSprites++;
    });
  });

  console.log("✅ Total sprites created:", totalSprites);
  console.log("Scene children count:", scene.children.length);
}

// --- NEURAL PATHS ---
function createNeuralPaths(sourceNode) {
  clearNeuralPaths();
  if (
    !linesEnabled ||
    sourceNode.userData.isNoise ||
    !sourceNode.visible ||
    !sourceNode.parent
  )
    return;

  const relatedNodes = clusterParticles.filter(
    (p) =>
      p.userData.cluster === sourceNode.userData.cluster &&
      p !== sourceNode &&
      p.visible &&
      p.parent
  );
  if (relatedNodes.length === 0) return;

  relatedNodes.forEach((targetNode, index) => {
    const curve = new THREE.CatmullRomCurve3([
      sourceNode.position.clone(),
      sourceNode.position
        .clone()
        .lerp(targetNode.position, 0.5)
        .add(
          new THREE.Vector3(
            (Math.random() - 0.5) * 10,
            (Math.random() - 0.5) * 10,
            (Math.random() - 0.5) * 10
          )
        ),
      targetNode.position.clone(),
    ]);
    const geometry = new THREE.BufferGeometry().setFromPoints(
      curve.getPoints(50)
    );
    const line = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({
        color: PALETTE.neural,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending,
      })
    );
    scene.add(line);
    neuralPaths.push(line);
    line.layers.enable(1);
    createPathParticle(curve, index * 0.2);
  });
}

function createPathParticle(curve, delay) {
  const particle = new THREE.Mesh(
    new THREE.SphereGeometry(0.8, 8, 8),
    new THREE.MeshBasicMaterial({
      color: PALETTE.neural,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    })
  );
  particle.userData = { curve, progress: -delay, speed: 0.3 };
  particle.layers.enable(1);
  scene.add(particle);
  pathParticles.push(particle);
}

function updatePathParticles(delta) {
  pathParticles.forEach((p, index) => {
    if (!p.userData.curve || !p.parent) {
      pathParticles.splice(index, 1);
      return;
    }

    p.userData.progress += delta * p.userData.speed;
    if (p.userData.progress > 1) p.userData.progress = 0;
    if (p.userData.progress >= 0 && p.userData.progress <= 1) {
      try {
        p.position.copy(p.userData.curve.getPoint(p.userData.progress));
        p.visible = true;
        p.scale.setScalar(1 + Math.sin(p.userData.progress * Math.PI) * 0.5);
      } catch (e) {
        p.visible = false;
      }
    } else p.visible = false;
  });
}

function clearNeuralPaths() {
  neuralPaths.forEach((l) => {
    l.geometry.dispose();
    l.material.dispose();
    scene.remove(l);
  });
  pathParticles.forEach((p) => {
    p.geometry.dispose();
    p.material.dispose();
    scene.remove(p);
  });
  neuralPaths = [];
  pathParticles = [];
}

// --- QUANTUM MERGE ---
function createBlackHole(position, color) {
  const core = new THREE.Mesh(
    new THREE.SphereGeometry(12, 32, 32),
    new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.95,
    })
  );

  const eventHorizon = new THREE.Mesh(
    new THREE.SphereGeometry(13, 32, 32),
    new THREE.MeshBasicMaterial({
      color: color,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
    })
  );

  const disk = new THREE.Mesh(
    new THREE.RingGeometry(15, 40, 64),
    new THREE.MeshBasicMaterial({
      color: color,
      transparent: true,
      opacity: 0.5,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    })
  );
  disk.rotation.x = Math.PI / 2;

  const innerRing = new THREE.Mesh(
    new THREE.TorusGeometry(16, 1.5, 16, 100),
    new THREE.MeshBasicMaterial({
      color: color,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
    })
  );

  const outerRing = new THREE.Mesh(
    new THREE.TorusGeometry(35, 2, 16, 100),
    new THREE.MeshBasicMaterial({
      color: color,
      transparent: true,
      opacity: 0.4,
      blending: THREE.AdditiveBlending,
    })
  );

  const particleGeometry = new THREE.BufferGeometry();
  const particleCount = 200;
  const positions = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount; i++) {
    const angle = (i / particleCount) * Math.PI * 2;
    const radius = 20 + Math.random() * 15;
    positions[i * 3] = Math.cos(angle) * radius;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 4;
    positions[i * 3 + 2] = Math.sin(angle) * radius;
  }
  particleGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(positions, 3)
  );
  const orbitParticles = new THREE.Points(
    particleGeometry,
    new THREE.PointsMaterial({
      color: color,
      size: 2,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    })
  );

  core.position.copy(position);
  eventHorizon.position.copy(position);
  innerRing.position.copy(position);
  outerRing.position.copy(position);
  disk.position.copy(position);
  orbitParticles.position.copy(position);

  const group = new THREE.Group();
  group.add(core, eventHorizon, innerRing, outerRing, disk, orbitParticles);
  scene.add(group);

  quantumMerge.blackHole = {
    group,
    core,
    eventHorizon,
    innerRing,
    outerRing,
    disk,
    orbitParticles,
    position: position.clone(),
    strength: 0,
    maxStrength: 4000,
    spawnTime: 0,
  };
}

function convertToParticles(sprite, count = 50) {
  if (!sprite) return;
  const particles = [];
  const spritePos = sprite.position.clone();
  const color = new THREE.Color(sprite.userData.originalColor);

  for (let i = 0; i < count; i++) {
    const p = new THREE.Mesh(
      new THREE.SphereGeometry(1.2, 4, 4),
      new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: 1,
        blending: THREE.AdditiveBlending,
      })
    );
    p.position.set(
      spritePos.x + (Math.random() - 0.5) * 15,
      spritePos.y + (Math.random() - 0.5) * 15,
      spritePos.z + (Math.random() - 0.5) * 15
    );

    p.userData = {
      velocity: new THREE.Vector3(
        Math.random() - 0.5,
        Math.random() - 0.5,
        Math.random() - 0.5
      ),
      life: 1.0,
      decayRate: 0.002 + Math.random() * 0.005,
    };
    p.layers.enable(1);
    scene.add(p);
    particles.push(p);
  }
  quantumMerge.particlePool.push(...particles);
}

function updateQuantumMerge(delta) {
  if (!quantumMerge.active) return;
  const bh = quantumMerge.blackHole;
  if (!bh) return;

  if (bh.spawnTime < 1) {
    bh.spawnTime += delta * 1.5;
    const t = Math.min(bh.spawnTime, 1);
    bh.strength = bh.maxStrength * t;
    bh.group.scale.setScalar(t);
  }

  const time = Date.now() * 0.001;
  if (bh.disk) {
    bh.disk.rotation.z += delta * 0.5;
  }
  if (bh.innerRing) {
    bh.innerRing.rotation.y += delta * 0.8;
    bh.innerRing.rotation.x = Math.PI / 2 + Math.sin(time) * 0.1;
  }
  if (bh.outerRing) {
    bh.outerRing.rotation.y -= delta * 0.4;
    bh.outerRing.rotation.x = Math.PI / 2 + Math.cos(time * 0.7) * 0.1;
  }
  if (bh.orbitParticles) {
    bh.orbitParticles.rotation.y += delta * 1.2;
  }

  if (bh.eventHorizon) {
    const pulse = Math.sin(time * 3) * 0.15 + 0.85;
    bh.eventHorizon.scale.setScalar(pulse);
    bh.eventHorizon.material.opacity = 0.4 + pulse * 0.2;
  }

  for (let i = quantumMerge.particlePool.length - 1; i >= 0; i--) {
    const p = quantumMerge.particlePool[i];
    if (p.userData.life <= 0) {
      scene.remove(p);
      p.geometry.dispose();
      p.material.dispose();
      quantumMerge.particlePool.splice(i, 1);
      continue;
    }

    const dir = new THREE.Vector3().subVectors(bh.position, p.position);
    const dist = dir.length();

    const forceMagnitude = (bh.strength*2) / (dist * dist + 10);
    dir.normalize().multiplyScalar(forceMagnitude * delta);

    const suctionBias = dir
      .clone()
      .normalize()
      .multiplyScalar(delta * 30);

    p.userData.velocity.add(dir);
    p.userData.velocity.add(suctionBias);

    p.userData.velocity.multiplyScalar(0.94);

    p.position.add(p.userData.velocity.clone().multiplyScalar(delta * 40));

    if (dist < 20) {
      p.userData.life -= 0.1;
      const stretch = Math.max(0.1, p.userData.life);
      p.scale.set(stretch * 0.3, stretch * 3, stretch * 0.3);
      p.lookAt(bh.position);
    } else if (dist < 50) {
      p.userData.life -= 0.02;
      p.rotation.x += delta * 5;
      p.rotation.y += delta * 3;
    } else {
      p.rotation.x += delta * 2;
      p.rotation.y += delta * 1.5;
    }

    p.material.opacity = p.userData.life;
  }
}

function finishQuantumMerge(callback) {
  if (!quantumMerge.blackHole) return;

  let t = 0;
  const animateFade = () => {
    if (t >= 1) {
      cleanupQuantumMerge();
      if (callback) callback();
      return;
    }
    t += 0.02;

    if (quantumMerge.blackHole) {
      quantumMerge.blackHole.group.traverse((obj) => {
        if (obj.material) {
          obj.material.opacity = Math.max(0, 1 - t);
        }
      });
    }

    quantumMerge.particlePool.forEach((p) => {
      p.material.opacity = Math.max(0, 1 - t);
    });

    requestAnimationFrame(animateFade);
  };
  animateFade();
}

function cleanupQuantumMerge() {
  if (quantumMerge.blackHole) {
    scene.remove(quantumMerge.blackHole.group);
    quantumMerge.blackHole.group.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    quantumMerge.blackHole = null;
  }
  quantumMerge.particlePool.forEach((p) => {
    scene.remove(p);
    p.geometry.dispose();
    p.material.dispose();
  });
  quantumMerge.particlePool = [];
  quantumMerge.active = false;
}

function performQuantumMerge(keepSprite, deleteSprites) {
  console.log("🌌 === QUANTUM MERGE STARTED ===");
  console.log("keepSprite:", keepSprite);
  console.log("deleteSprites count:", deleteSprites?.length);

  if (!keepSprite || quantumMerge.active) {
    console.warn("⚠️ Cannot start: keepSprite missing or merge already active");
    return;
  }

  if (hoveredNode) {
    clearNeuralPaths();
    hoveredNode = null;
  }

  quantumMerge.active = true;
  console.log("✅ Quantum merge activated");

  console.log("📹 Flying to position:", keepSprite.position);
  flyTo(keepSprite.position, 250);

  console.log("⚫ Creating black hole...");
  createBlackHole(keepSprite.position, keepSprite.userData.originalColor);

  console.log("💥 Converting sprites to particles...");
  deleteSprites.forEach((sprite) => {
    if (sprite && sprite.parent) {
      convertToParticles(sprite, 30);
      sprite.visible = false;
      console.log("  - Converted:", sprite.userData.filename);
    }
  });

  console.log("⏰ Setting 3 second timeout for cleanup...");
  setTimeout(() => {
    console.log("🧹 Starting cleanup...");
    finishQuantumMerge(() => {
      console.log("🗑️ Removing sprites from scene...");
      deleteSprites.forEach((sprite) => {
        if (sprite && sprite.parent) {
          scene.remove(sprite);
          sprite.material.dispose();
          if (sprite.geometry) sprite.geometry.dispose();
          const index = clusterParticles.indexOf(sprite);
          if (index > -1) clusterParticles.splice(index, 1);
        }
      });

      document.body.style.cursor = "default";
      window.dispatchEvent(new CustomEvent("universe-unhover"));

      console.log("📢 Dispatching quantum-merge-complete event");
      window.dispatchEvent(
        new CustomEvent("quantum-merge-complete", { detail: { deleteSprites } })
      );

      console.log("✅ === QUANTUM MERGE COMPLETE ===");
    });
  }, 3000);
}

// --- INTERACTION ---

function onMouseClick(event, callback) {
  if (quantumMerge.active) return;

  const validSprites = clusterParticles.filter((s) => s.visible && s.parent);
  const intersects = raycaster.intersectObjects(validSprites);

  if (intersects.length > 0) {
    const target = intersects[0].object;
    if (target.userData.isNoise) {
      flyTo(target.position);
      return;
    }
    if (callback) callback(target.userData);
  }
}

function animate() {
  requestAnimationFrame(animate);
  const delta = Math.min(clock.getDelta(), 0.1);
  const time = clock.getElapsedTime();

  // Pulsating particles with more dramatic effect
  clusterParticles.forEach((sprite) => {
    if (!sprite.visible || !sprite.parent) return;
    const data = sprite.userData;

    if (!quantumMerge.active && sprite.material.opacity > 0.3) {
      // More dramatic pulsation
      const pulse = 1 + Math.sin(time * 2 + sprite.id * 0.1) * 0.25;
      sprite.scale.setScalar(data.originalScale * pulse);

      // Subtle color shifting for non-noise particles
      if (!data.isNoise) {
        const brightness = 1 + Math.sin(time * 1.5 + sprite.id * 0.2) * 0.2;
        sprite.material.opacity = Math.min(1.0, 0.8 + brightness * 0.2);
      }
    }
  });

  activeLines.forEach((l) => {
    if (l.visible && l.material.opacity < 0.4)
      l.material.opacity += delta * 0.5;
  });

  updatePathParticles(delta);
  updateQuantumMerge(delta);
  controls.update();
  composer.render();
}

function flyTo(target, dist = 60) {
  const startPos = camera.position.clone();
  const startTarget = controls.target.clone();
  const dir = new THREE.Vector3().subVectors(startPos, startTarget).normalize();
  const endPos = target.clone().add(dir.multiplyScalar(dist));
  let p = 0;
  function step() {
    p += 0.02;
    if (p > 1) p = 1;
    const e = 1 - Math.pow(1 - p, 3);
    camera.position.lerpVectors(startPos, endPos, e);
    controls.target.lerpVectors(startTarget, target, e);
    if (p < 1) requestAnimationFrame(step);
  }
  step();
}

function onWindowResize() {
  const c = renderer.domElement.parentElement;
  if (!c) return;
  camera.aspect = c.clientWidth / c.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(c.clientWidth, c.clientHeight);
  composer.setSize(c.clientWidth, c.clientHeight);
  console.log("🔄 Universe resized to:", c.clientWidth, "x", c.clientHeight);
}

function onPointerMove(e) {
  if (quantumMerge.active) return;

  const r = renderer.domElement.getBoundingClientRect();
  pointer.x = ((e.clientX - r.left) / r.width) * 2 - 1;
  pointer.y = -((e.clientY - r.top) / r.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  const validSprites = clusterParticles.filter((s) => s.visible && s.parent);
  const hits = raycaster.intersectObjects(validSprites);

  if (hits.length > 0) {
    const obj = hits[0].object;
    if (obj !== hoveredNode && !obj.userData.isNoise) {
      hoveredNode = obj;
      if (linesEnabled) createNeuralPaths(obj);
    }
    document.body.style.cursor = "pointer";
    window.dispatchEvent(
      new CustomEvent("universe-hover", { detail: obj.userData })
    );
  } else {
    if (hoveredNode) {
      clearNeuralPaths();
      hoveredNode = null;
    }
    document.body.style.cursor = "default";
    window.dispatchEvent(new CustomEvent("universe-unhover"));
  }
}
