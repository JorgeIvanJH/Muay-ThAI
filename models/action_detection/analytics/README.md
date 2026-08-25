# Analytics: from frame predictions to useful measurements

The classifiers answer “what label best describes this frame?” Analytics
answers event-level questions:

- Was this motion one completed strike?
- Which anatomical limb most likely performed it?
- How many strikes have occurred?
- What was the approximate endpoint speed?

These calculations are deterministic. They do not train another model and can
be improved without retraining TCN or LightGBM.

## Where analytics sits in the system

```text
selected raw YOLO keypoints ------+
                                  |
                                  v
                         scale + kinematics
                                  |
striking class probabilities -----+
                                  |
                                  v
                         event state machine
                                  |
                         +--------+--------+
                         |                 |
                         v                 v
                       counts        speed summary
                         |                 |
                         +--------+--------+
                                  v
                      overlay, CSV and JSON
```

The normalized classifier features are camera-resilient, but physical speed
needs raw pixel displacement. That is why analytics receives both the selected
raw keypoints and the striking probabilities.

## Files in this folder

| File | Responsibility |
|---|---|
| `types.py` | Small shared records: `JointPoint`, `SpeedSample`, `StrikeEvent`. |
| `anthropometry.py` | Converts supplied height and projected bones into pixels per metre. |
| `kinematics.py` | Smooths keypoints and calculates timestamp-based endpoint speeds. |
| `strike_events.py` | Confirms, delimits and de-duplicates limb-specific events. |
| `strike_speed.py` | Summarizes sampled speed and optionally refines the peak. |
| `pipeline.py` | Connects all stages, maintains counts, draws text and writes outputs. |
| `__init__.py` | Exposes the small public API used by shared inference. |

The main entry point for other code is `StrikeAnalytics` from `pipeline.py`.

## Shared data records in `types.py`

### `JointPoint`

One raw image-plane point:

```text
x_px, y_px, confidence
```

### `SpeedSample`

One physical speed estimate paired with its video timestamp:

```text
timestamp, speed_mps
```

### `StrikeEvent`

A completed event contains:

- numeric event ID;
- strike type and anatomical side;
- start, apex and end frames;
- corresponding timestamps;
- peak classification confidence;
- valid speed samples collected during the event.

Keeping this record immutable prevents a completed result changing later.

## Anthropometric scale

Pixels are not metres. `AnthropometricScaleEstimator` uses the user-supplied
height and approximate adult long-bone proportions:

| Segment | Fraction of height used |
|---|---:|
| upper arm | 0.186 |
| forearm | 0.146 |
| thigh | 0.245 |
| lower leg | 0.246 |

For every confident visible segment:

```text
pixels per metre = projected segment length in pixels
                   ----------------------------------
                   estimated real segment length
```

A bone appears shorter when it points partly towards the camera. The estimator
therefore keeps a short history and uses a high projected-length quantile for
each segment. It then takes the median across visible segments and smooths the
session scale with an exponential moving average.

This is robust enough for an educational side-view estimate, but it is not an
individual anatomical measurement. Body proportions vary, and monocular video
cannot observe depth motion.

The inference CLI uses 175 cm by default:

```powershell
--person-height-cm 182
```

## Kinematics

`KinematicsTracker` processes one timestamped pose at a time.

### Causal smoothing

Raw keypoints jitter even when a person is stationary. An exponential moving
average combines the new point with its previous smoothed position. The
default alpha is 0.5: responsive enough for strikes while reducing some pose
noise.

Because this is causal smoothing, it never reads future frames and can operate
live. Smoothing also attenuates very short velocity peaks, which is one reason
speed remains approximate.

### Velocity

For each valid joint:

```text
pixel speed = distance(current point, previous point) / elapsed video time
speed m/s   = pixel speed / pixels per metre
```

Actual video timestamps are used rather than inference wall-clock time. A file
can take several minutes to process without changing the physical duration of
its motion.

Velocity is not calculated across a gap longer than 0.12 seconds. This prevents
a joint that disappears and later reappears from producing an enormous false
speed.

### Strike endpoints

The basic implementation associates each class with one endpoint:

| Strike | Left endpoint | Right endpoint |
|---|---|---|
| punch | left wrist | right wrist |
| elbow | left elbow | right elbow |
| kick | left ankle | right ankle |
| knee | left knee | right knee |

During the candidate phase, the faster eligible side is selected and then
locked for the event. Locking prevents the assigned limb switching halfway
through a strike because of one noisy frame.

## The strike event state machine

Frame classifications should not be counted directly. A label can last many
frames, flicker briefly, or remain confident across two nearby movements.

The current states are:

```text
IDLE -> CANDIDATE -> ACTIVE -> IDLE
          |                    |
          +---- rejected ------+
```

### Initial rules

`IDLE -> CANDIDATE` requires:

- the strongest non-background strike probability to reach 0.60;
- a valid corresponding limb endpoint;
- endpoint speed above its strike-specific threshold;
- the same strike/side not to be in cooldown.

`CANDIDATE -> ACTIVE` requires the same strike and side for two frames. A
one-frame probability or motion spike is therefore rejected.

An `ACTIVE` event remains supported while:

- its class probability is at least 0.35; and
- its locked limb maintains 45% of the activation motion threshold.

A 0.10-second release grace tolerates brief keypoint or classification loss.
Valid events must last at least 0.08 seconds, are forced to finish after 2
seconds and give the same type/side a 0.15-second cooldown.

Default activation speeds are:

| Strike | Threshold |
|---|---:|
| punch | 0.80 m/s |
| elbow | 0.60 m/s |
| kick | 0.90 m/s |
| knee | 0.60 m/s |

These are initial engineering values, not learned truths. They should be tuned
against event-level ground truth from unseen videos.

### Important current limitation

The state machine uses speed magnitude. Magnitude has no direction, so a fast
retraction can resemble a second strike—particularly for elbows. The most
valuable next improvement is a strike-specific phase model such as:

```text
ready -> offensive motion -> apex -> recovery -> ready
```

That model should use signed extension, joint angles or endpoint distance from
the relevant shoulder/hip. Longer cooldowns can hide some duplicate counts but
may also suppress legitimate combinations, so threshold-only tuning is not a
complete solution.

## Speed summaries and polynomial refinement

`estimate_strike_speed` reports:

| Field | Meaning |
|---|---|
| `average_mps` | Mean of valid event speed samples. |
| `sampled_peak_mps` | Largest speed actually observed at 30 FPS. |
| `robust_peak_mps` | Mean of the three fastest valid samples. |
| `interpolated_peak_mps` | Guarded local quadratic estimate. |
| `interpolated_peak_timestamp` | Sub-frame timestamp of that fitted peak. |

The robust peak is less sensitive to one noisy keypoint. The interpolated peak
fits a quadratic to up to five samples around the sampled maximum. It is used
only when:

- at least three unique timestamps exist;
- the parabola is concave;
- its vertex lies inside the local sample interval;
- it is no more than 50% above the sampled peak.

Otherwise the sampled peak is returned. A polynomial can estimate a smooth
peak between frames, but it cannot reconstruct motion that 30 FPS never
captured. Higher-degree polynomials are intentionally avoided because they can
oscillate and invent implausible peaks.

## `pipeline.py`: the public coordinator

`AnalyticsConfig` chooses:

```text
enabled metrics
person height
keypoint confidence threshold
smoothing alpha
```

It is a typed Python dataclass rather than an external configuration file
because these values are created directly from validated CLI arguments and are
used by only one runtime component. Algorithm-specific state-machine thresholds
remain beside the state machine in `StrikeStateMachineConfig`. If later
experiments require many named threshold profiles, a YAML/JSON experiment
configuration would become worthwhile.

For every frame, `StrikeAnalytics.update`:

1. Removes invalid or low-confidence raw points.
2. Updates scale, smoothing and endpoint velocities.
3. Supplies probabilities and motion to the state machine.
4. Converts completed events into optional speed estimates.
5. Updates limb-specific counts.
6. Returns an `AnalyticsSnapshot` for JSONL and the overlay.

At end-of-stream, `finalize` flushes a valid active event. `write_outputs`
writes event rows and the aggregate summary.

## Selecting metrics

The shared inference parser supports:

```powershell
--metrics count
--metrics speed
--metrics count speed
```

Both are enabled by default. Even speed-only mode still needs the event state
machine to determine which trajectory belongs to one strike.

## Saved outputs

### Events CSV

Each row contains:

```text
event_id, strike_type, side,
start/apex/end frame and time,
duration, peak classification confidence,
average, sampled, robust and interpolated speed,
valid speed sample count
```

This is the best file for event-level evaluation and error inspection.

### Summary JSON

The summary records:

- enabled metrics and person height;
- smoothing and confidence settings;
- every state-machine threshold;
- total and limb-specific counts;
- per-strike average and maximum robust speed;
- final pixels-per-metre estimate.

### Predictions JSONL

The shared frame record includes current analytics state, active strike,
current counts and any event completed on that frame. JSONL is useful for
plotting classifier probability and state transitions over time.

## How to validate and improve the counter

Total counts alone can hide one missed event and one duplicate. Prefer one row
per ground-truth strike:

```csv
video_id,event_id,strike_type,side,start_frame,apex_frame,end_frame
video_01,1,punch,left,125,132,141
```

If full boundaries are expensive, type, side and apex frame are already useful.

Use development videos to tune rules and untouched videos for final reporting.
Match predicted and true events within a frame tolerance, then report:

- event precision, recall and F1;
- count error by strike and limb;
- boundary/apex timing error;
- failure counts from missing joints or wrong side assignment.

## How to validate speed

Counts do not validate physical speed. Useful references include:

- high-frame-rate video at the same viewpoint;
- known image-plane motion over a measured distance;
- radar or another instrumented reference.

A practical experiment is to record at 120 FPS, treat that as the reference,
downsample to 30 FPS and compare sampled, robust and interpolated peaks.

## Suggested exercises

1. Plot left/right endpoint speed and class probability around each event.
2. Replace magnitude-only punch motion with shoulder-to-wrist extension rate.
3. Add an explicit recovery state and evaluate duplicate elbows.
4. Compare EMA alphas on jitter and peak attenuation.
5. Evaluate scale from arms only, legs only and all available segments.
6. Measure the polynomial's benefit on downsampled high-frame-rate recordings.
