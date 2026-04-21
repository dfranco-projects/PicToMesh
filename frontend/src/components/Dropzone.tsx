import { useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface Props {
  onFiles: (files: File[]) => void
  disabled?: boolean
}

const ACCEPTED = { "image/*": [".jpg", ".jpeg", ".png", ".webp"] }

export function Dropzone({ onFiles, disabled }: Props) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) onFiles(accepted)
    },
    [onFiles],
  )

  const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    disabled,
    multiple: true,
  })

  return (
    <div
      {...getRootProps()}
      className={cn(
        "relative flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 text-center transition-colors cursor-pointer select-none",
        isDragActive
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/50 hover:bg-muted/30",
        disabled && "pointer-events-none opacity-50",
      )}
    >
      <input {...getInputProps()} />

      <div className="flex flex-col items-center gap-2">
        <UploadIcon />
        <p className="text-sm font-medium text-foreground">
          {isDragActive ? "Drop images here" : "Drag & drop images"}
        </p>
        <p className="text-xs text-muted-foreground">
          or <span className="text-primary underline underline-offset-2">browse files</span>
        </p>
        <p className="text-xs text-muted-foreground">JPG, PNG, WEBP · multiple views recommended</p>
      </div>

      {acceptedFiles.length > 0 && (
        <Badge variant="secondary" className="mt-1">
          {acceptedFiles.length} {acceptedFiles.length === 1 ? "image" : "images"} selected
        </Badge>
      )}
    </div>
  )
}

function UploadIcon() {
  return (
    <svg
      className="h-10 w-10 text-muted-foreground"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
      />
    </svg>
  )
}
