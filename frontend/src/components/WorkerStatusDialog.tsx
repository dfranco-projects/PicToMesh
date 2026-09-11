import { useRef } from "react"
import { CircleAlert, ExternalLink, LoaderCircle } from "lucide-react"
import type { WorkerError } from "@/lib/api"
import type { WorkerView } from "@/hooks/useWorkerStatus"
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { buttonVariants } from "@/components/ui/button"

const NEW_ISSUE_URL = "https://github.com/dfranco-projects/PicToMesh/issues/new"
const MAX_DETAIL_CHARS = 2000 // keeps the prefilled issue URL within browser limits

interface Copy {
  title: string
  description: string
  hint: string
  error?: WorkerError
  retrying?: boolean
}

function copyFor(view: WorkerView): Copy | null {
  switch (view.kind) {
    case "failed":
      return {
        title: "PicToMesh couldn't load its AI models",
        description: view.error.message,
        hint: "Fix the problem and restart PicToMesh. This window closes by itself once the models are ready.",
        error: view.error,
      }
    case "retrying":
      return {
        title: "PicToMesh couldn't load its AI models",
        description: view.error.message,
        hint: "Trying again now. This window closes by itself once the models are ready.",
        error: view.error,
        retrying: true,
      }
    case "offline":
      return {
        title: "The processing worker isn't running",
        description: "PicToMesh's background worker stopped or never started, so images can't be processed.",
        hint: "Restart PicToMesh. This window closes by itself once the worker is back.",
      }
    case "unreachable":
      return {
        title: "Can't reach PicToMesh",
        description: "The PicToMesh server isn't responding.",
        hint: "Make sure PicToMesh is running. This page reconnects by itself.",
      }
    default:
      return null
  }
}

function issueUrl({ title, description, error }: Copy): string {
  const body = [
    "**What happened**",
    description,
    "",
    "**Technical details**",
    "```",
    (error?.detail ?? "none").slice(0, MAX_DETAIL_CHARS),
    "```",
    "",
    "**How I run PicToMesh** (make local or docker):",
    "",
    "**Operating system:**",
  ].join("\n")
  return `${NEW_ISSUE_URL}?${new URLSearchParams({ title, body })}`
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
}

/** Blocks the app while the worker can't take jobs; closes itself once it can. */
export function WorkerStatusDialog({ view }: { view: WorkerView }) {
  const copy = copyFor(view)
  const reportRef = useRef<HTMLAnchorElement>(null)

  return (
    <AlertDialog open={copy !== null}>
      {copy && (
        <AlertDialogContent className="data-[size=default]:sm:max-w-md" initialFocus={reportRef}>
          {/* Fixed columns: with auto ones, short copy lets the icon column grow. */}
          <AlertDialogHeader className="sm:grid-cols-[auto_1fr]">
            <AlertDialogMedia className="bg-destructive/10 text-destructive">
              <CircleAlert />
            </AlertDialogMedia>
            <AlertDialogTitle>{copy.title}</AlertDialogTitle>
            <AlertDialogDescription>{copy.description}</AlertDialogDescription>
          </AlertDialogHeader>

          <div className="flex flex-col gap-3 text-sm text-muted-foreground">
            <p className="flex items-center gap-2">
              {copy.retrying && <LoaderCircle className="size-4 animate-spin" />}
              {copy.hint}
            </p>
            {copy.error && (
              <>
                <p className="text-xs">Last attempt at {formatTime(copy.error.at)}</p>
                <details className="rounded-lg border border-border bg-muted/40 px-3 py-2">
                  <summary className="cursor-pointer text-xs font-medium text-foreground">
                    Technical details
                  </summary>
                  <pre className="mt-2 max-h-48 overflow-auto text-xs whitespace-pre-wrap break-words">
                    {copy.error.detail}
                  </pre>
                </details>
              </>
            )}
          </div>

          <AlertDialogFooter>
            <a
              ref={reportRef}
              href={issueUrl(copy)}
              target="_blank"
              rel="noreferrer"
              className={buttonVariants({ variant: "outline" })}
            >
              Report an issue
              <ExternalLink />
            </a>
          </AlertDialogFooter>
        </AlertDialogContent>
      )}
    </AlertDialog>
  )
}
