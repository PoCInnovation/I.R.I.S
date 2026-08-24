# PimEyes Reverse Search — How it works


## 1. What it does, in one sentence

Given a cropped photo of a face, it uploads that photo to **PimEyes**
(a public face-search engine) exactly the way a human would in a browser, gets
past the anti-bot protections, and launches the search.

The public entry point is one function:

```python
await pimeyes_reverse_search(face_crop)   # face_crop is a BGR image (NumPy array)
```

> **Note on results:** PimEyes hides the actual matching photos behind a paywall.
> Extracting them is out of scope here, so the function currently always returns
> an **empty list** (`[]`) and just logs the search-page URL. It is written so
> that URL extraction can be added later without changing the rest of the flow.

---

## 2. Why this is hard (the problem we're solving)

PimEyes does **not** want bots automating searches. It defends itself with:

- A **CAPTCHA** ("I am human" checkbox, made by a company called *Prosopo*).
- **Bot fingerprinting**: it inspects the browser for tell-tale signs of
  automation (e.g. "headless" browsers that run with no visible window).

A naive script that opens a browser and clicks buttons gets detected and blocked
immediately. So this module's whole job is to **look like a real person on a real
computer**, while still running automatically on a server with no screen.

---

## 3. The tools involved (glossary)

| Tool | What it is | Why we use it |
|------|-----------|---------------|
| **Playwright** | A library to control a web browser from code (click, type, upload…). | Automates the PimEyes page. |
| **Camoufox** | A specially-hardened build of Firefox + Playwright, made to *not look like a bot*. It can fake the operating system, spoof location, and move the mouse in a human, curved way. | This is the browser we drive. It hides the automation fingerprints PimEyes looks for. |
| **Xvfb** | A "virtual screen" for Linux servers that have no monitor. The browser draws onto it, but nobody sees it. | Lets us run a *real* (non-headless) browser invisibly on a server. This matters: a truly headless browser is easy to detect, but a real browser on a fake screen is not. |
| **x11vnc + a VNC viewer** | Screen-sharing tools. `x11vnc` broadcasts the virtual screen; a "VNC viewer" is a window that lets you see and click on it. | If the CAPTCHA needs a human, we pop open a window showing the exact live browser session so a person can click it. |

You need these installed on Linux:

```bash
sudo apt install xvfb x11vnc
sudo apt install tigervnc-viewer   # provides a VNC viewer (xtigervncviewer)
```

If any are missing, the code degrades gracefully (see §6) instead of crashing.

---

## 4. The key idea: headless, but not *really* headless

There are three ways to run a browser:

1. **Visible** — a real window you can see. Undetectable, but useless on a
   headless server and annoying for automation.
2. **Headless** — no window at all. Convenient, but PimEyes **detects it** and
   blocks the search.
3. **Our approach** — a *real, fully-rendered* browser (`headless=False`) that
   draws onto an **invisible virtual screen (Xvfb)**.

Option 3 gives us the best of both: it runs invisibly on a server, but from
PimEyes' point of view it's an ordinary Firefox with an ordinary screen, mouse,
and rendering — no headless fingerprint.

The **human-like mouse movement** (`humanize`) reinforces this. Prosopo scores
*how* the cursor moves before you click. A single dead-straight line to the
checkbox looks robotic, so the code deliberately drifts the cursor around
first (`_wander_cursor`) and moves in human-paced curves.

---

## 5. The flow, step by step

When you call `pimeyes_reverse_search(face_crop)`:

1. **Save the photo to disk.** PimEyes' upload needs a real file, so the face
   crop is written to a temporary `.png`.

2. **Start the invisible screen.** A `VirtualDisplay` (Xvfb) is created; we note
   its display number (e.g. `:99`) because we may need it later for VNC.

3. **Launch Camoufox** onto that display, then in `_run_search`:
   - **Open PimEyes** (`https://pimeyes.com/en`).
   - **Upload the face** by setting the file directly on the page's hidden file
     input (`_upload`).
   - **Accept consent.** Uploading opens a pop-up (modal) with three checkboxes
     (18+, Terms, Privacy). They must all be ticked or the search button stays
     greyed out (`_accept_consent`).
   - **Clear the CAPTCHA automatically** (`_auto_solve_prosopo`): wander the
     cursor, then click the real "I am human" checkbox. If that doesn't work, it
     tries a few slightly different click positions.
   - **If the CAPTCHA still isn't solved, hand off to a human**
     (`_human_captcha_handoff`) — see §6.
   - **Press "Start Search"** (`_start_search`) and wait for the browser to land
     on the results page (URL contains `/search/`).

4. **Clean up.** Delete the temp photo and shut down the virtual screen, always,
   even if something failed.

### How we know the CAPTCHA is solved

We don't try to read Prosopo's internal state. Instead we watch the **"Start
Search" button**: PimEyes keeps it disabled until the CAPTCHA is cleared. The
moment it becomes clickable (`_start_enabled`), we know we're through — no matter
whether a script or a human clicked the checkbox.

---

## 6. When a human is needed (the VNC handoff)

Sometimes Prosopo escalates and demands a real person. Because the browser is
running invisibly, we can't just "show" it — an open browser window is tied to
its screen for its whole life and can't be moved.

So instead we **make the invisible screen viewable, in place**:

1. `x11vnc` starts broadcasting the virtual screen over the local machine only.
2. A **VNC viewer** window opens showing the *exact same live PimEyes page* —
   already uploaded, already on the CAPTCHA.
3. A human clicks the checkbox in that window.
4. As soon as "Start Search" enables (`_wait_for_human`), the script continues
   automatically. No relaunch, no re-upload.
5. The viewer and `x11vnc` are shut down.

This whole handoff has a time limit (`manual_solve_timeout_ms`, default 3
minutes). If nobody solves it in time, the search is abandoned cleanly.

**Graceful degradation:** if Xvfb isn't installed, the browser just runs in a
normal visible window instead. If `x11vnc` or a VNC viewer isn't installed, the
code prints the address to connect to manually instead of crashing.

---

## 7. Function parameters

```python
await pimeyes_reverse_search(
    face_crop,                       # BGR image array (e.g. Face.crop from FaceDetector)
    max_results=10,                  # cap on URLs (currently unused — results are paywalled)
    timeout_ms=60_000,               # per-step browser timeout
    manual_solve_timeout_ms=180_000, # how long to wait for a human on the CAPTCHA
    headless=True,                   # True: invisible + VNC-on-demand; False: visible window (debugging)
)
```

Set `headless=False` when debugging locally to watch the whole thing happen in a
real window.

---

## 8. Failure handling & debugging

- **Nothing ever crashes the pipeline.** Any failure — a changed button, a
  blocked search, a timeout — is caught and turned into an empty list `[]`. The
  wider I.R.I.S pipeline treats that as "no match" and moves to the next face.
- **On failure, a screenshot is saved** to `/tmp/iris_pimeyes_failure.png`
  (`_dump_failure`). This is the first thing to look at when a search stops
  working — it usually shows whether we were stuck on the CAPTCHA, the consent
  boxes, or a page-layout change.
- **Progress is logged** to the console with a `[pimeyes]` prefix at each step.

---

## 9. Why the selectors look strange

PimEyes' page is built so that its buttons have almost no stable "names" in the
HTML, which makes them hard to target reliably. The code leans on the few stable
hooks that exist:

- The consent boxes are `<button role="checkbox">`, **not** real checkboxes, so
  standard checkbox selectors don't match them.
- The upload control is a hidden file input; we set the file on it directly.
- Prosopo's checkbox is found via a `data-cy` attribute (a testing hook that
  survives redesigns), with a fallback that clicks several offsets because the
  checkbox shifts position depending on the page's language.

If PimEyes redesigns its site, these selectors are the most likely things to
need updating — start from the failure screenshot.
