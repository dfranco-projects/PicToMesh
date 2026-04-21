import { Suspense, useEffect, useState } from "react"
import { Canvas } from "@react-three/fiber"
import { OrbitControls, Center, Environment, useGLTF } from "@react-three/drei"

interface Props {
  url: string
}

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url)
  return (
    <Center>
      <primitive object={scene} />
    </Center>
  )
}

export function Viewer3D({ url }: Props) {
  // Bust GLTF cache on new URL
  const [key, setKey] = useState(0)
  useEffect(() => {
    useGLTF.preload(url)
    setKey((k) => k + 1)
  }, [url])

  return (
    <div className="relative h-full w-full rounded-xl overflow-hidden bg-muted/20">
      <Canvas
        key={key}
        camera={{ position: [0, 0, 3], fov: 45 }}
        gl={{ antialias: true }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={1} />
        <Suspense fallback={null}>
          <Model url={url} />
          <Environment preset="city" />
        </Suspense>
        <OrbitControls makeDefault autoRotate autoRotateSpeed={0.8} />
      </Canvas>
      <p className="absolute bottom-3 left-1/2 -translate-x-1/2 text-xs text-muted-foreground pointer-events-none select-none">
        Drag to rotate · Scroll to zoom
      </p>
    </div>
  )
}
