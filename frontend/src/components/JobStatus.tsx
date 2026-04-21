import { useEffect, useRef, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { fetchJob, type JobResult } from "@/lib/api"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"

interface Props {
  jobId: string
  onComplete: (result: JobResult) => void
}

interface ProgressEvent {
  status: string
  message: string
  progress: number
}

export function JobStatus({ jobId, onComplete }: Props) {
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState("Queued…")
  const doneRef = useRef(false)

  // SSE stream for live progress
  useEffect(() => {
    const es = new EventSource(`/jobs/${jobId}/stream`)
    es.onmessage = (e) => {
      try {
        const ev: ProgressEvent = JSON.parse(e.data)
        setMessage(ev.message || ev.status)
        setProgress(Math.round(ev.progress * 100))
        if (ev.status === "complete" || ev.status === "failed") {
          es.close()
        }
      } catch {}
    }
    es.onerror = () => es.close()
    return () => es.close()
  }, [jobId])

  // Poll as fallback / source of truth for completion
  const { data } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => fetchJob(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === "complete" || status === "failed" ? false : 2000
    },
  })

  useEffect(() => {
    if (data && !doneRef.current) {
      if (data.status === "complete" || data.status === "failed") {
        doneRef.current = true
        if (data.status === "complete") setProgress(100)
        onComplete(data)
      }
    }
  }, [data, onComplete])

  const isFailed = data?.status === "failed"

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between text-sm">
        <span className={cn("text-muted-foreground", isFailed && "text-destructive")}>
          {isFailed ? data?.error ?? "Processing failed" : message}
        </span>
        <span className="tabular-nums text-xs text-muted-foreground">{progress}%</span>
      </div>
      <Progress value={progress} className={cn(isFailed && "[&>div]:bg-destructive")} />
    </div>
  )
}
