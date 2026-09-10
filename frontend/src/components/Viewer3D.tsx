import { Suspense, useEffect, useState } from "react"
import { Canvas } from "@react-three/fiber"
import { OrbitControls, Center, Environment, useGLTF } from "@react-three/drei"
import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react"

interface Props {
  url: string
}

const SPEED_MIN = 0.2
const SPEED_MAX = 5
const SPEED_FACTOR = 1.5

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

  const [rotating, setRotating] = useState(true)
  const [speed, setSpeed] = useState(0.8)
  const canSlow = rotating && speed > SPEED_MIN
  const canSpeed = rotating && speed < SPEED_MAX

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
        <OrbitControls makeDefault autoRotate={rotating} autoRotateSpeed={speed} />
      </Canvas>
      <div className="absolute bottom-9 left-1/2 -translate-x-1/2 flex items-center gap-5 text-muted-foreground">
        <button
          type="button"
          onClick={() => setSpeed((s) => Math.max(SPEED_MIN, s / SPEED_FACTOR))}
          disabled={!canSlow}
          aria-label="Rotate slower"
          title="Slower"
          className="transition-colors hover:text-foreground disabled:opacity-30 disabled:hover:text-muted-foreground"
        >
          <ChevronLeft className="size-4" />
        </button>
        <button
          type="button"
          onClick={() => setRotating((r) => !r)}
          aria-label={rotating ? "Pause rotation" : "Resume rotation"}
          title={rotating ? "Pause" : "Play"}
          className="transition-colors hover:text-foreground"
        >
          {rotating ? <Pause className="size-4 fill-current" /> : <Play className="size-4 fill-current" />}
        </button>
        <button
          type="button"
          onClick={() => setSpeed((s) => Math.min(SPEED_MAX, s * SPEED_FACTOR))}
          disabled={!canSpeed}
          aria-label="Rotate faster"
          title="Faster"
          className="transition-colors hover:text-foreground disabled:opacity-30 disabled:hover:text-muted-foreground"
        >
          <ChevronRight className="size-4" />
        </button>
      </div>
      <p className="absolute bottom-3 left-1/2 -translate-x-1/2 text-xs text-muted-foreground pointer-events-none select-none">
        Drag to rotate · Scroll to zoom
      </p>
    </div>
  )
}
