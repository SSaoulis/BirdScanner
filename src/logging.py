import logging
def on_track_became_stable(track):
    # Species may be unknown until classification runs; log what we have.
    logging.debug(
        "Track became stable: track_id=%s species=%s box=%s stable_frames=%s",
        track.track_id,
        track.species,
        track.box,
        track.stable_frames,
    )

def on_track_deleted(track):
    logging.debug(
        "Track deleted: track_id=%s species=%s box=%s stable_frames=%s missing_frames=%s",
        track.track_id,
        track.species,
        track.box,
        track.stable_frames,
        track.frames_since_seen,
    )