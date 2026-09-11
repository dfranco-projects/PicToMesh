import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { LoaderCircle } from "lucide-react"
import { Dropzone } from "@/components/Dropzone"
import { JobStatus } from "@/components/JobStatus"
import { Viewer3D } from "@/components/Viewer3D"
import { ThemeToggle } from "@/components/ThemeToggle"
import { WorkerStatusDialog } from "@/components/WorkerStatusDialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { submitJob, type JobResult } from "@/lib/api"
import { useStore } from "@/store/useStore"
import { useWorkerStatus } from "@/hooks/useWorkerStatus"

type Stage = "idle" | "processing" | "done" | "error"

export default function App() {
  const [files, setFiles] = useState<File[]>([])
  const [fmt, setFmt] = useState<"glb" | "obj" | "stl">("glb")
  const [stage, setStage] = useState<Stage>("idle")
  const [error, setError] = useState<string | null>(null)
  const { jobId, meshUrl, setJobId, setMeshUrl, reset } = useStore()
  const worker = useWorkerStatus()
  const workerReady = worker.kind === "ready"
  const preparing = worker.kind === "starting" || worker.kind === "loading"

  function fail(reason?: string | null) {
    setError(reason?.trim() || null)
    setStage("error")
  }

  const { mutate: startJob, isPending } = useMutation({
    mutationFn: () => submitJob(files, fmt),
    onSuccess: (data) => {
      setJobId(data.job_id)
      setStage("processing")
    },
    onError: (e: Error) => fail(e.message),
  })

  function handleComplete(result: JobResult) {
    if (result.status === "complete" && result.mesh_url) {
      setMeshUrl(result.mesh_url)
      setStage("done")
    } else {
      // The worker's reason reaches us here; JobStatus unmounts on error, so
      // without keeping it the user only ever sees a generic failure.
      fail(result.error)
    }
  }

  function handleReset() {
    reset()
    setFiles([])
    setError(null)
    setStage("idle")
  }

  const canSubmit = files.length > 0 && stage === "idle" && workerReady

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      {/* ── Left panel ── */}
      <aside className="flex w-[360px] shrink-0 flex-col gap-6 border-r border-border p-6">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">PicToMesh</h1>
            <p className="text-xs text-muted-foreground mt-0.5">Images → 3D mesh</p>
          </div>
          <ThemeToggle />
        </div>

        <Separator />

        {/* Dropzone */}
        <Dropzone onFiles={setFiles} disabled={stage !== "idle"} />
        <p className="text-xs text-muted-foreground -mt-3">
          Best results: one object on a plain background, filling the frame. More photos taken
          around the object give a better mesh as long as neighbouring angles overlap; a single
          photo falls back to a model that guesses the unseen sides.
        </p>

        {/* Format picker */}
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground shrink-0">Output format</span>
          <Select
            value={fmt}
            onValueChange={(v) => setFmt(v as typeof fmt)}
            disabled={stage !== "idle"}
          >
            <SelectTrigger className="h-8 flex-1 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="glb">GLB</SelectItem>
              <SelectItem value="obj">OBJ</SelectItem>
              <SelectItem value="stl">STL</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Progress */}
        {(stage === "processing" || stage === "done") && jobId && (
          <JobStatus jobId={jobId} onComplete={handleComplete} />
        )}

        {stage === "error" && (
          <div className="flex flex-col gap-1">
            <p className="text-sm text-destructive">Something went wrong. Please try again.</p>
            {error && (
              <p className="text-xs text-muted-foreground break-words">{error}</p>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="mt-auto flex flex-col gap-2">
          {stage === "idle" && preparing && (
            <p className="flex items-start gap-2 text-xs text-muted-foreground">
              <LoaderCircle className="mt-px size-3.5 shrink-0 animate-spin" />
              {worker.kind === "loading"
                ? "Preparing the AI models. The first start downloads about 5 GB, so this can take a while."
                : "Starting PicToMesh…"}
            </p>
          )}

          {stage === "idle" && (
            <Button onClick={() => startJob()} disabled={!canSubmit || isPending} className="w-full">
              {isPending ? "Submitting…" : preparing ? "Preparing models…" : "Generate mesh"}
            </Button>
          )}

          {stage === "done" && meshUrl && (
            <>
              <a href={meshUrl} download>
                <Button className="w-full">Download {fmt.toUpperCase()}</Button>
              </a>
              <Button variant="outline" className="w-full" onClick={handleReset}>
                Start over
              </Button>
            </>
          )}

          {(stage === "processing" || stage === "error") && (
            <Button variant="outline" className="w-full" onClick={handleReset}>
              Cancel
            </Button>
          )}
        </div>

        {/* Footer badges */}
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs text-muted-foreground">
            {files.length > 0 ? `${files.length} image${files.length > 1 ? "s" : ""}` : "No files"}
          </Badge>
          {files.length >= 5 && (
            <Badge variant="outline" className="text-xs text-muted-foreground">
              Multi-view mode
            </Badge>
          )}
        </div>
      </aside>

      {/* ── Right panel — 3D viewer ── */}
      <main className="flex flex-1 flex-col items-center justify-center p-6">
        {stage === "done" && meshUrl ? (
          <Viewer3D url={meshUrl} />
        ) : (
          <EmptyViewer stage={stage} />
        )}
      </main>

      <WorkerStatusDialog view={worker} />
    </div>
  )
}

function EmptyViewer({ stage }: { stage: Stage }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed border-border w-full h-full text-center">
      {stage === "processing" ? (
        <>
          <SpinnerIcon />
          <p className="text-sm text-muted-foreground">Generating your 3D mesh…</p>
        </>
      ) : (
        <>
          <CubeIcon />
          <p className="text-sm text-muted-foreground">Your 3D mesh will appear here</p>
          <p className="text-xs text-muted-foreground">Upload images and click Generate</p>
        </>
      )}
    </div>
  )
}

function CubeIcon() {
  return (
    <svg
      className="h-12 w-12 text-muted-foreground/40"
      fill="none"
      stroke="currentColor"
      strokeWidth={1}
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9"
      />
    </svg>
  )
}

function SpinnerIcon() {
  return (
    <svg className="h-10 w-10 animate-spin text-primary" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  )
}
