import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { CropEditor } from "../components/CropEditor";

type Tab = "test" | "region";

/**
 * Camera page.
 *
 * Two tabs:
 *  - "Test" captures an on-demand snapshot of the current (cropped) detection
 *    feed to confirm the camera works.
 *  - "Detection region" hosts the interactive crop editor, where the user drags
 *    a box over a full-sensor preview to set what the object detector sees.
 *
 * Snapshots are fetched as blobs so HTTP errors (e.g. the detector being
 * offline) surface as a readable message rather than a broken-image icon.
 * Object URLs are revoked when replaced or on unmount to avoid leaks.
 */
export function Camera() {
  const [tab, setTab] = useState<Tab>("test");
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  // Revoke any outstanding object URL when the component unmounts.
  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  async function testCamera() {
    setLoading(true);
    setError(null);
    try {
      // Cache-bust so each click fetches a genuinely fresh frame.
      const res = await fetch(`${api.camera.snapshotUrl()}?t=${Date.now()}`);
      if (!res.ok) {
        throw new Error(`Camera returned ${res.status} ${res.statusText}`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = url;
      setImageSrc(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to capture image");
    } finally {
      setLoading(false);
    }
  }

  const tabClass = (active: boolean): string =>
    `px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
      active ? "bg-emerald-600" : "bg-slate-700 hover:bg-slate-600"
    }`;

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Camera</h1>
        <div className="flex gap-2">
          <button onClick={() => setTab("test")} className={tabClass(tab === "test")}>
            Test
          </button>
          <button
            onClick={() => setTab("region")}
            className={tabClass(tab === "region")}
          >
            Detection region
          </button>
        </div>
      </div>

      {tab === "test" ? (
        <>
          <div className="flex items-center justify-between gap-3">
            <p className="text-sm text-slate-400">
              Capture a live frame straight from the camera to confirm it is
              working.
            </p>
            <button
              onClick={testCamera}
              disabled={loading}
              className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-semibold transition-colors"
            >
              {loading ? "Capturing…" : "Test Camera"}
            </button>
          </div>

          {error && (
            <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-300">
              {error}
            </div>
          )}

          <div className="rounded-2xl bg-slate-800 p-4 flex items-center justify-center min-h-[20rem]">
            {imageSrc ? (
              <img
                src={imageSrc}
                alt="Latest camera capture"
                className="max-h-[70vh] w-auto rounded-lg"
              />
            ) : (
              <p className="text-sm text-slate-500">
                {loading
                  ? "Capturing…"
                  : 'No image yet — press "Test Camera" to capture one.'}
              </p>
            )}
          </div>
        </>
      ) : (
        <CropEditor />
      )}
    </div>
  );
}
