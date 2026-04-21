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

export async function submitJob(files: File[], fmt = "glb"): Promise<JobResponse> {
  const form = new FormData()
  files.forEach((f) => form.append("files", f))
  const res = await fetch(`/jobs?fmt=${fmt}`, { method: "POST", body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function fetchJob(jobId: string): Promise<JobResult> {
  const res = await fetch(`/jobs/${jobId}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export function meshDownloadUrl(meshUrl: string, fmt: string): string {
  // meshUrl is already a relative path like /meshes/{id}/mesh.glb
  // For different format, swap the extension
  return meshUrl.replace(/\.\w+$/, `.${fmt}`)
}
