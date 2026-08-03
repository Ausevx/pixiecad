import type React from "react";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { usePageVisible, useReducedMotion } from "@/lib/hooks";
import { formatCount } from "@/lib/api";

// Traverse and dispose a material and all attached textures to prevent WebGL memory leaks
function disposeMaterial(mat: THREE.Material): void {
  mat.dispose();
  for (const key of Object.keys(mat)) {
    const value = (mat as unknown as Record<string, unknown>)[key];
    if (
      value &&
      typeof value === "object" &&
      "isTexture" in value &&
      typeof (value as THREE.Texture).dispose === "function"
    ) {
      (value as THREE.Texture).dispose();
    }
  }
}

export default function ModelViewer(props: {
  url: string;
  className?: string;
}): React.JSX.Element {
  const { url, className } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<boolean>(false);
  // Distinguished from a load failure: there is nothing to retry and no
  // point offering to, so it gets its own honest message.
  const [noWebgl, setNoWebgl] = useState<boolean>(false);
  const [triangleCount, setTriangleCount] = useState<number | null>(null);

  const isPageVisible = usePageVisible();
  const isReducedMotion = useReducedMotion();

  const isIntersectingRef = useRef<boolean>(true);
  const isPageVisibleRef = useRef<boolean>(isPageVisible);
  const isReducedMotionRef = useRef<boolean>(isReducedMotion);
  const controlsRef = useRef<OrbitControls | null>(null);
  const startAnimationRef = useRef<(() => void) | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  isPageVisibleRef.current = isPageVisible;
  isReducedMotionRef.current = isReducedMotion;

  // Update OrbitControls autoRotate when reduced motion setting changes
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = !isReducedMotion;
    }
  }, [isReducedMotion]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    setLoading(true);
    setError(false);
    setTriangleCount(null);

    // 1. Initialize Three.js Scene & Renderer
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#06070a");

    const camera = new THREE.PerspectiveCamera(
      45,
      container.clientWidth / Math.max(container.clientHeight, 1),
      0.1,
      1000
    );

    // WebGL is not guaranteed: a blocklisted driver, a headless or remote
    // session, or a browser with it disabled all throw here. Constructing the
    // renderer outside a guard let that exception escape the effect and take
    // the entire application down with it, leaving a blank page instead of a
    // job whose artifacts were sitting right there and downloadable.
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      setNoWebgl(true);
      setLoading(false);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight, false);
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";
    container.appendChild(renderer.domElement);

    // 2. Lighting setup: Hemisphere light + single Directional light (no shadows for performance)
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x333333, 1.2);
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight.position.set(5, 10, 7.5);
    scene.add(dirLight);

    // 3. Orbit Controls setup
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = !isReducedMotionRef.current;
    controls.autoRotateSpeed = 1.0;
    controlsRef.current = controls;

    // 4. On-demand rendering & visibility controls
    const render = () => {
      if (!isIntersectingRef.current || !isPageVisibleRef.current) {
        animFrameIdRef.current = null;
        return;
      }
      controls.update();
      renderer.render(scene, camera);
      animFrameIdRef.current = requestAnimationFrame(render);
    };

    const startAnimation = () => {
      if (
        animFrameIdRef.current === null &&
        isIntersectingRef.current &&
        isPageVisibleRef.current
      ) {
        animFrameIdRef.current = requestAnimationFrame(render);
      }
    };

    const stopAnimation = () => {
      if (animFrameIdRef.current !== null) {
        cancelAnimationFrame(animFrameIdRef.current);
        animFrameIdRef.current = null;
      }
    };

    startAnimationRef.current = startAnimation;

    // 5. Container ResizeObserver & Viewport IntersectionObserver
    const resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      if (width === 0 || height === 0) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
      if (
        animFrameIdRef.current === null &&
        isIntersectingRef.current &&
        isPageVisibleRef.current
      ) {
        renderer.render(scene, camera);
      }
    });
    resizeObserver.observe(container);

    const intersectionObserver = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (entry) {
          isIntersectingRef.current = entry.isIntersecting;
          if (entry.isIntersecting) {
            startAnimation();
          } else {
            stopAnimation();
          }
        }
      },
      { threshold: 0.01 }
    );
    intersectionObserver.observe(container);

    // Initial check for visible state
    if (isIntersectingRef.current && isPageVisibleRef.current) {
      startAnimation();
    }

    // 6. Load GLB Model
    let isCancelled = false;
    let loadedModel: THREE.Group | null = null;
    const loader = new GLTFLoader();

    loader.load(
      url,
      (gltf) => {
        if (isCancelled) {
          gltf.scene.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
              const mesh = child as THREE.Mesh;
              mesh.geometry?.dispose();
              if (Array.isArray(mesh.material)) {
                mesh.material.forEach(disposeMaterial);
              } else if (mesh.material) {
                disposeMaterial(mesh.material);
              }
            }
          });
          return;
        }

        loadedModel = gltf.scene;
        scene.add(loadedModel);

        // Compute total triangle count
        let tris = 0;
        loadedModel.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            const geom = mesh.geometry;
            if (geom) {
              if (geom.index) {
                tris += geom.index.count / 3;
              } else if (geom.attributes.position) {
                tris += geom.attributes.position.count / 3;
              }
            }
          }
        });
        setTriangleCount(Math.round(tris));

        // Auto-frame bounding box
        const box = new THREE.Box3().setFromObject(loadedModel);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const fovRad = camera.fov * (Math.PI / 180);
        let dist = maxDim > 0 ? (maxDim / (2 * Math.tan(fovRad / 2))) * 1.5 : 5;

        controls.target.copy(center);
        camera.position.set(
          center.x + dist * 0.5,
          center.y + dist * 0.5,
          center.z + dist
        );
        camera.near = Math.max(0.01, dist / 100);
        camera.far = Math.max(1000, dist * 100);
        camera.updateProjectionMatrix();
        controls.update();

        setLoading(false);
        renderer.render(scene, camera);
      },
      undefined,
      () => {
        if (isCancelled) return;
        setLoading(false);
        setError(true);
      }
    );

    // Dispose all resources on unmount or URL change
    return () => {
      isCancelled = true;
      stopAnimation();
      startAnimationRef.current = null;
      controlsRef.current = null;

      resizeObserver.disconnect();
      intersectionObserver.disconnect();

      scene.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          mesh.geometry?.dispose();
          if (Array.isArray(mesh.material)) {
            mesh.material.forEach(disposeMaterial);
          } else if (mesh.material) {
            disposeMaterial(mesh.material);
          }
        }
      });

      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    };
  }, [url]);

  // Restart rendering when returning to tab
  useEffect(() => {
    if (isPageVisible && isIntersectingRef.current) {
      startAnimationRef.current?.();
    }
  }, [isPageVisible]);

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden bg-void ${className ?? ""}`}
      role="img"
      aria-label="3D model preview"
    >
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center font-mono text-xs text-ink-faint">
          loading model…
        </div>
      )}
      {noWebgl && (
        <div className="absolute inset-0 flex items-center justify-center p-4 text-center">
          <p className="font-mono text-[11px] leading-relaxed text-ink-faint">
            3D preview unavailable — this browser has no WebGL.
            <br />
            The model and its parts are still downloadable above.
          </p>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center font-mono text-xs text-fail">
          failed to load model
        </div>
      )}
      {!loading && !error && triangleCount !== null && (
        <div className="absolute bottom-2 right-2 rounded-sharp border border-rule bg-panel/80 px-2 py-1 font-mono text-[11px] text-ink-dim backdrop-blur-sm pointer-events-none select-none">
          {formatCount(triangleCount)} tris
        </div>
      )}
    </div>
  );
}
