import { useQuery } from "@tanstack/react-query"
import { fetchWorkerStatus, type WorkerError, type WorkerStatus } from "@/lib/api"

export type WorkerView =
  | { kind: "ready" }
  | { kind: "starting" } // no word from the worker yet
  | { kind: "loading" }
  | { kind: "retrying"; error: WorkerError }
  | { kind: "failed"; error: WorkerError }
  | { kind: "offline" }
  | { kind: "unreachable" }

// The worker only reports in once its imports finish (torch, and numba's
// compile on a first run), so silence is normal for a while after a start.
const GRACE_MS = 60_000

interface Poll {
  status: WorkerStatus | null // null: the API itself didn't answer
  overdue: boolean // silent for longer than GRACE_MS
}

// When the worker or the API went silent. Kept outside React because it is
// observed by the query function, not during render.
let silentSince: number | null = null

async function poll(): Promise<Poll> {
  const now = Date.now()
  let status: WorkerStatus | null = null
  try {
    status = await fetchWorkerStatus()
  } catch {
    // Reported as unreachable below.
  }
  const silent = status === null || status.state === "offline"
  silentSince = silent ? (silentSince ?? now) : null
  return { status, overdue: silentSince !== null && now - silentSince >= GRACE_MS }
}

function toView({ status, overdue }: Poll): WorkerView {
  if (status === null) return overdue ? { kind: "unreachable" } : { kind: "starting" }
  switch (status.state) {
    case "ready":
      return { kind: "ready" }
    case "loading":
      return status.error ? { kind: "retrying", error: status.error } : { kind: "loading" }
    case "failed":
      return { kind: "failed", error: status.error }
    case "offline":
      return overdue ? { kind: "offline" } : { kind: "starting" }
  }
}

/** Whether the worker can take jobs, polled faster while it can't. */
export function useWorkerStatus(): WorkerView {
  const { data } = useQuery({
    queryKey: ["worker-status"],
    queryFn: poll,
    refetchInterval: (query) => (query.state.data?.status?.state === "ready" ? 10_000 : 3_000),
    refetchIntervalInBackground: true,
  })
  return data ? toView(data) : { kind: "starting" }
}
