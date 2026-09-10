"use client";

import Image from "next/image";
import { Suspense, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Center, Environment, Html, OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";

function Model({ paused, focused, onClick }: { paused: boolean; focused: boolean; onClick: () => void }) {
  const { scene } = useGLTF("/models/aircraft-engine.glb");
  const group = useRef<THREE.Group>(null);
  useFrame((_, delta) => { if (group.current && !paused) group.current.rotation.y += delta * 0.7; });
  return <group ref={group} scale={2.4} rotation={[0, -0.4, 0]} onClick={(event) => { event.stopPropagation(); onClick(); }}><Center><primitive object={scene.clone()} /></Center>{focused && <Html center position={[0, 1.7, 0]}><div className="engine-tooltip">Aircraft piston engine<small>Focused component model</small></div></Html>}</group>;
}
function Diagram(){return <div className="engine-schematic-image"><Image src="/engine/uav-engine-schematic.png" alt="Labeled UAV piston-engine technical schematic" fill priority unoptimized quality={100} sizes="(max-width: 700px) 100vw, 620px" /></div>}
export function EngineTwin() {
  const [mode, setMode] = useState<"3D" | "2D">("3D");
  const [paused, setPaused] = useState(false);
  const [focused, setFocused] = useState(false);
  return <div className="engine-twin engine-twin-model">
    <div className="engine-twin-top"><span>INTERACTIVE UAV PISTON ENGINE</span><small>{paused ? "Rotation paused" : "360° model · drag to orbit · scroll to zoom"}</small></div>
    {mode === "3D" ? <Canvas camera={{ position: [3.6, 1.9, 5.2], fov: 30 }} dpr={[1, 1.5]}><color attach="background" args={["#071526"]} /><ambientLight intensity={1.25} /><directionalLight position={[5, 6, 5]} intensity={4} color="#d8f5ff" /><pointLight position={[-4, 2, 3]} intensity={15} color="#22c9ff" /><pointLight position={[2, -3, 1]} intensity={4} color="#ffb13b" /><Suspense fallback={<Html center><div className="engine-loading">Loading 3D engine…</div></Html>}><Model paused={paused} focused={focused} onClick={() => setFocused((value) => !value)} /><Environment preset="city" /></Suspense><OrbitControls enablePan={false} minDistance={2.8} maxDistance={10} /></Canvas> : <Diagram />}
    <div className="engine-callout intake">Air intake</div><div className="engine-callout propeller">Propeller</div><div className="engine-callout cylinders">Cylinder bank</div><div className="engine-callout fuel">Fuel injection</div>
    <div className="engine-twin-label lubrication">● Lubrication<br /><small>Early Degradation</small></div><div className="engine-twin-label injector">● Injector<br /><small>Normal</small></div>
    <div className="engine-twin-controls"><button className={mode === "3D" ? "active" : ""} onClick={() => setMode("3D")}>3D</button><button className={mode === "2D" ? "active" : ""} onClick={() => setMode("2D")}>2D</button><button className="engine-rotate-toggle" onClick={() => setPaused((value) => !value)}>{paused ? "Play rotation" : "Pause rotation"}</button></div>
  </div>;
}
useGLTF.preload("/models/aircraft-engine.glb");
