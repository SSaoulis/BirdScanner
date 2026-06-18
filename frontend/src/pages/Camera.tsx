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
      active ? "bg-gold text-card shadow-sm" : "border border-line bg-card text-bark hover:text-ink"
    }`;

  return (
    <div className="mx-auto max-w-3xl px-6 py-8 space-y-6">
      <div className="flex items-end justify-between gap-4">
        <header>
          <p className="eyebrow mb-2">The lens</p>
          <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">Camera</h1>
        </header>
        <div className="flex gap-2">
          <button onClick={() => setTab("test")} className={tabClass(tab === "test")}>
            Test shot
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
            <p className="text-sm text-bark">
              Grab a single frame from the camera to check it&rsquo;s pointed at the
              feeder and seeing clearly.
            </p>
            <button
              onClick={testCamera}
              disabled={loading}
              className="rounded-lg bg-gold px-4 py-2 text-sm font-semibold text-card shadow-sm transition-colors hover:bg-gold-deep disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Capturing…" : "Take a test shot"}
            </button>
          </div>

          {error && (
            <div className="rounded-lg border border-rust/40 bg-rust/10 px-4 py-3 text-sm text-rust">
              {error}
            </div>
          )}

          <div className="flex min-h-[20rem] items-center justify-center rounded-2xl border border-line bg-card p-4 shadow-plate">
            {imageSrc ? (
              <img
                src={imageSrc}
                alt="Latest camera capture"
                className="max-h-[70vh] w-auto rounded-lg shadow-plate"
              />
            ) : (
              <p className="text-sm text-bark">
                {loading
                  ? "Capturing…"
                  : "No shot yet — press “Take a test shot” to see what the camera sees."}
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
