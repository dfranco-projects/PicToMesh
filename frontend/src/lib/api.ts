export type JobStatus = "queued" | "in_progress" | "complete" | "failed"

export interface JobResponse {
  job_id: string
  status: JobStatus
}

export interface JobResult {
  job_id: string
  status: JobStatus
  mesh_url: string | null
  error: string | null
}

/** A failed API call, with a message fit to show the user. */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

// FastAPI puts a sentence in `detail`, or a list of field errors for a 422.
async function errorMessage(res: Response): Promise<string> {
  try {
    const { detail } = await res.json()
    if (typeof detail === "string") return detail
    if (Array.isArray(detail)) return detail.map((d: { msg: string }) => d.msg).join(" ")
  } catch {
    // Not a FastAPI body, e.g. a proxy's HTML error page.
  }
  return `The server returned an unexpected error (HTTP ${res.status}).`
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch {
    throw new ApiError(0, "Can't reach the PicToMesh server.")
  }
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res))
  return res.json()
}

export function submitJob(files: File[], fmt = "glb"): Promise<JobResponse> {
  const form = new FormData()
  files.forEach((f) => form.append("files", f))
  return request(`/jobs?fmt=${fmt}`, { method: "POST", body: form })
}

export function fetchJob(jobId: string): Promise<JobResult> {
  return request(`/jobs/${jobId}`)
}

export function meshDownloadUrl(meshUrl: string, fmt: string): string {
  // meshUrl is already a relative path like /meshes/{id}/mesh.glb
  // For different format, swap the extension
  return meshUrl.replace(/\.\w+$/, `.${fmt}`)
}
