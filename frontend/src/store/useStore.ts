import { create } from "zustand"

interface AppState {
  jobId: string | null
  meshUrl: string | null
  setJobId: (id: string | null) => void
  setMeshUrl: (url: string | null) => void
  reset: () => void
}

export const useStore = create<AppState>((set) => ({
  jobId: null,
  meshUrl: null,
  setJobId: (id) => set({ jobId: id, meshUrl: null }),
  setMeshUrl: (url) => set({ meshUrl: url }),
  reset: () => set({ jobId: null, meshUrl: null }),
}))
