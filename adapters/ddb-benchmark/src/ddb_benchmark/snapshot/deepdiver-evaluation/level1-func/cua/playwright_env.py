"""
Playwright browser environment for the CUA.

Manages browser lifecycle, provides screenshot capture,
page navigation, and DOM inspection primitives.

Also captures structured runtime signals (console errors,
pageerrors, failed network requests, 4xx/5xx responses,
external hosts hit, click hit-tests) that downstream
verifiers need to evaluate ``data_integration``,
``runtime_health`` and ``interaction_accessibility`` claims.
"""
from __future__ import annotations
import asyncio
import base64
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from cua import net_trace

logger = logging.getLogger(__name__)

VISIBLE_TEXT_MAX_CHARS = 8000

# Repo root, used to resolve repo-relative fixture paths (e.g.
# ``data/mock_resources/...``) independently of the process CWD.
# .../DDB_evaluation-main/deepdiver-evaluation/level1-func/cua/playwright_env.py
REPO_DIR = Path(__file__).resolve().parents[3]

# Built-in asset roots for seed batches whose fixture paths are relative to an
# external folder (e.g. "S1-011/cat_a_avatar.png"). Extend/override at runtime
# with the FIXTURES_DIR env var (':'-separated list, searched first).
DEFAULT_FIXTURE_ROOTS = [
    REPO_DIR / "data/CN_query_1-3_0710/stage0-3_20260711_cn_0-100queries",
    REPO_DIR / "data/EN_query_1-3_0710/stage0-3_20260711_en_0-80queries",
]

# ── Virtual clock shim ──────────────────────────────────────────
# Injected via context.add_init_script BEFORE any page script runs, so the
# app builds its timers/Date reads on top of our controllable clock. The
# shim is INERT until ``__advanceVirtualClock(ms)`` is called: while the
# offset is 0 it behaves like real time (a single shared driver timer wakes
# at the soonest virtual deadline == the real delay), so normal apps run
# unchanged. Calling advance jumps the offset forward and synchronously
# fires every timer whose virtual deadline has passed — letting the CUA
# "fast-forward" a 25-min countdown to completion in milliseconds without
# the harness having to wait 25 real minutes.
#
# It overrides: setTimeout / setInterval / clearTimeout / clearInterval /
# requestAnimationFrame / cancelAnimationFrame / Date / Date.now /
# performance.now. Covers both timer styles seen in generated apps:
#   * counter-based (decrement a JS var each setInterval tick) — fast-fired
#   * Date-diff-based (remaining = endTime - Date.now()) — offset makes the
#     next tick compute the completed state.
_VIRTUAL_CLOCK_SHIM = r"""
(() => {
  if (window.__virtualClockInstalled) return;
  window.__virtualClockInstalled = true;

  const RealDate = Date;
  const origSetTimeout = window.setTimeout.bind(window);
  const origClearTimeout = window.clearTimeout.bind(window);
  const origPerfNow = (window.performance && window.performance.now)
    ? window.performance.now.bind(window.performance)
    : (() => RealDate.now());

  let offset = 0; // virtual ms ahead of real time
  const realNow = () => RealDate.now();
  const virtualNow = () => realNow() + offset;

  // ── scheduler (one shared real driver timer) ────────────────
  const timers = new Map(); // id -> {cb, due, interval, args}
  let nextId = 1;
  let driverId = null;

  function soonestDue() {
    let m = Infinity;
    for (const t of timers.values()) if (t.due < m) m = t.due;
    return m;
  }

  function reschedule() {
    if (driverId !== null) { origClearTimeout(driverId); driverId = null; }
    if (timers.size === 0) return;
    const delay = Math.max(0, soonestDue() - virtualNow());
    driverId = origSetTimeout(drive, delay);
  }

  function drive() {
    driverId = null;
    const MAX_FIRE = 500000;
    const wallStart = realNow();
    let fired = 0;
    while (true) {
      const now = virtualNow();
      let pick = null, pickId = -1;
      for (const [id, t] of timers) {
        if (t.due <= now && (pick === null || t.due < pick.due)) { pick = t; pickId = id; }
      }
      if (pick === null) break;
      if (pick.interval > 0) {
        // Advance by exactly one interval and let the loop re-fire it on
        // the next pass while still due. We deliberately do NOT snap/skip
        // missed ticks: counter-based countdowns decrement a variable on
        // every tick, so each one must actually run. MAX_FIRE + the
        // wall-time guard below bound the total work.
        pick.due += pick.interval;
      } else {
        timers.delete(pickId);
      }
      try { pick.cb.apply(null, pick.args || []); } catch (e) {}
      fired++;
      if (fired >= MAX_FIRE) break;
      if (realNow() - wallStart > 8000) break; // wall-time safety guard
    }
    reschedule();
  }

  function addTimer(cb, delay, interval, args) {
    const id = nextId++;
    timers.set(id, {
      cb,
      due: virtualNow() + Math.max(0, Number(delay) || 0),
      interval: interval || 0,
      args,
    });
    reschedule();
    return id;
  }

  window.setTimeout = function (cb, delay, ...args) {
    if (typeof cb !== 'function') return origSetTimeout(cb, delay, ...args);
    return addTimer(cb, delay, 0, args);
  };
  window.setInterval = function (cb, delay, ...args) {
    if (typeof cb !== 'function') return origSetTimeout(cb, delay, ...args);
    return addTimer(cb, delay, Math.max(1, Number(delay) || 1), args);
  };
  window.clearTimeout = function (id) {
    if (timers.has(id)) { timers.delete(id); reschedule(); }
    else origClearTimeout(id);
  };
  window.clearInterval = window.clearTimeout;

  window.requestAnimationFrame = function (cb) {
    return addTimer(function () { cb(origPerfNow() + offset); }, 16, 0, []);
  };
  window.cancelAnimationFrame = window.clearTimeout;

  // ── Date override ───────────────────────────────────────────
  function FakeDate(...args) {
    if (!(this instanceof FakeDate)) return new RealDate(virtualNow()).toString();
    if (args.length === 0) return new RealDate(virtualNow());
    return new RealDate(...args);
  }
  FakeDate.prototype = RealDate.prototype;
  FakeDate.now = () => virtualNow();
  FakeDate.parse = RealDate.parse;
  FakeDate.UTC = RealDate.UTC;
  try { Object.setPrototypeOf(FakeDate, RealDate); } catch (e) {}
  window.Date = FakeDate;

  // ── performance.now override (offset-adjusted, monotonic) ────
  if (window.performance) {
    try {
      Object.defineProperty(window.performance, 'now', {
        configurable: true,
        value: () => origPerfNow() + offset,
      });
    } catch (e) {}
  }

  // ── public control surface ──────────────────────────────────
  window.__advanceVirtualClock = function (ms) {
    offset += Math.max(0, Number(ms) || 0);
    drive();
    return { offset_ms: offset, virtual_now: virtualNow(), pending_timers: timers.size };
  };
  window.__getVirtualClock = function () {
    return { offset_ms: offset, virtual_now: virtualNow(), pending_timers: timers.size };
  };
  window.__resetVirtualClock = function () {
    offset = 0; reschedule();
    return { offset_ms: 0, pending_timers: timers.size };
  };
})();
"""

# ── TTS (speechSynthesis) shim ──────────────────────────────────
# Injected via context.add_init_script BEFORE any page script runs, so
# every call the app makes to ``speechSynthesis.speak()`` is recorded in
# ``window.__ttsLog`` with the utterance text/lang/voice. Browser TTS is
# otherwise invisible to the harness: it produces no screenshot change,
# no network request and no console output, which made TTS claims (e.g.
# "点击小喇叭后朗读该句日文") land on ``insufficient_evidence``.
#
# The shim records the ``speak()`` CALL itself (works even when the
# headless engine has no voices and never actually vocalises) and
# best-effort updates ``started`` / ``ended`` / ``error`` flags from the
# utterance lifecycle events. Each page load gets a fresh random ``gen``
# token so the Python side can de-duplicate entries across reloads.
_TTS_SHIM = r"""
(() => {
  if (window.__ttsShimInstalled) return;
  window.__ttsShimInstalled = true;
  window.__ttsLog = [];
  window.__ttsLogGen =
    String(Date.now()) + '-' + Math.random().toString(36).slice(2, 8);

  const ss = window.speechSynthesis;
  if (!ss || !window.SpeechSynthesisUtterance) return;

  const origSpeak = ss.speak.bind(ss);
  ss.speak = function (utterance) {
    try {
      if (window.__ttsLog.length < 100) {
        const entry = {
          event: 'speak',
          text: String((utterance && utterance.text) || '').slice(0, 300),
          lang: (utterance && utterance.lang) || '',
          voice: (utterance && utterance.voice && utterance.voice.name) || '',
          ts: Date.now(),
          started: false,
          ended: false,
          error: null,
        };
        window.__ttsLog.push(entry);
        if (utterance && typeof utterance.addEventListener === 'function') {
          utterance.addEventListener('start', () => { entry.started = true; });
          utterance.addEventListener('end', () => { entry.ended = true; });
          utterance.addEventListener('error', (e) => {
            entry.error = (e && e.error) ? String(e.error) : 'error';
          });
        }
      }
    } catch (e) {}
    return origSpeak(utterance);
  };
})();
"""

# Hard ceiling for a single advance_time call (24h). Prevents a runaway
# request from driving millions of interval ticks.
MAX_ADVANCE_MS = 24 * 60 * 60 * 1000

# Caps on how many runtime entries we keep per claim. Overflows are
# silently dropped after a single warning so we never blow up the
# downstream prompt payload.
MAX_CONSOLE_ERRORS = 50
MAX_FAILED_REQUESTS = 80
MAX_EXTERNAL_HOSTS = 40
MAX_HIT_TESTS = 30
# Runtime audit channel: every backend/API call (success AND failure) is
# recorded here, separate from ``_failed_requests`` (which feeds the
# page-load-failure detector and only keeps 4xx/5xx). For 4xx/5xx we also
# fetch the response body to extract the PostgreSQL / PostgREST error code.
MAX_BACKEND_CALLS = 200
BACKEND_BODY_MAX_CHARS = 1500
# TTS shim channel: speak() calls recorded per claim (see _TTS_SHIM).
MAX_TTS_EVENTS = 50

# URL substrings that mark a request as a backend/API call even when it is
# served from the same origin as the app (e.g. a proxy that mounts Supabase
# under ``/rest/v1`` on the preview host). Cross-origin calls are detected
# separately via host comparison.
_API_PATH_MARKERS = (
    "/rest/v1/", "/auth/v1/", "/storage/v1/", "/functions/v1/",
    "/rpc/", "/api/", "supabase",
)

# Hosts that are noisy and not interesting for data_integration scoring.
# We still surface them in console/network logs if they error, but they
# do NOT count as "external API actually called" evidence.
_DEV_HOSTS_BLOCKLIST = {
    "localhost", "127.0.0.1", "0.0.0.0",
}


def _now_ms() -> float:
    return time.time() * 1000.0


def _truncate(text: str, n: int) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def _host_of(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
    except Exception:
        return ""
    return (parsed.hostname or "").lower()


class PlaywrightEnv:
    """Manages one Playwright browser instance for CUA interaction."""

    def __init__(
        self,
        viewport: dict[str, int] | None = None,
        headless: bool = True,
        enable_virtual_clock: bool = True,
        fixture_roots: list[str | Path] | None = None,
    ) -> None:
        self._viewport = viewport or {"width": 1280, "height": 900}
        self._headless = headless
        # Extra asset roots for resolving relative fixture paths (from the
        # ``fixtures_dirs`` config key). Searched before the built-in
        # DEFAULT_FIXTURE_ROOTS; the FIXTURES_DIR env var overrides both.
        self._fixture_roots: list[Path] = [Path(r) for r in fixture_roots or []]
        # When True, a controllable virtual-clock shim is injected before
        # any page script runs. Inert until ``advance_clock`` is called, so
        # normal apps behave as if on real time. Lets the CUA fast-forward
        # long countdowns/timers instead of waiting wall-clock minutes.
        self._enable_virtual_clock = enable_virtual_clock
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

        # ── Multi-session support (0702) ────────────────────────────
        # A child env spawned by ``spawn_isolated_session`` shares the parent's
        # Browser but owns a fresh, isolated BrowserContext (its own cookies /
        # localStorage / anonymous identity). It must NOT tear down the shared
        # Browser / Playwright on ``stop``; ``_owns_browser`` guards that.
        # The parent tracks live children so ``stop`` can close any leftovers.
        self._owns_browser: bool = True
        self._child_sessions: list["PlaywrightEnv"] = []

        # Preview deployments gate access behind a Cloudflare challenge (403)
        # unless requests carry the ``X-Preview-Test`` header. We inject it
        # ONLY for same-origin requests to the preview host (set in ``goto``);
        # attaching it to cross-origin CDNs (map tiles, fonts) would trigger a
        # CORS preflight those hosts reject, silently breaking the map.
        # Prefer env; fall back to the shared eval secret so health checks /
        # CUA don't need ``export PREVIEW_TEST_SECRET=...`` every session.
        # (Do NOT use context ``extra_http_headers`` for this — same CORS pitfall.)
        self._preview_secret = (
            os.environ.get("PREVIEW_TEST_SECRET", "").strip()
            or ""
        )
        self._preview_host: str = ""

        self._console_errors: list[dict[str, Any]] = []
        self._failed_requests: list[dict[str, Any]] = []
        self._backend_calls: list[dict[str, Any]] = []
        self._external_hosts: dict[str, dict[str, Any]] = {}
        self._hit_tests: list[dict[str, Any]] = []
        self._runtime_started_at_ms: float = 0.0

        # TTS evidence buffer. The authoritative copy lives here (page-side
        # ``window.__ttsLog`` is wiped on navigation); ``_tts_index`` maps
        # (gen, idx) -> the buffered entry so late lifecycle updates
        # (started/ended/error) from the page can be merged in-place and
        # reloads never double-count.
        self._tts_events: list[dict[str, Any]] = []
        self._tts_index: dict[tuple[str, int], dict[str, Any]] = {}

        # Last mouse position recorded by any of the mouse primitives below.
        # The Anthropic native ``cursor_position`` action expects an actual
        # coordinate back; previously we returned a screenshot which left
        # Claude with no positional answer (see __cursor__ sentinel in
        # ``native_adapter.py``). Updating this field inside every mouse
        # primitive keeps the cached value honest without an extra
        # Playwright round-trip.
        self._last_cursor_xy: tuple[int, int] = (0, 0)

        # ── File-upload fixtures (harness capability) ───────────────
        # Per-query files the harness will attach to ANY native file-picker
        # the page opens (an OS dialog we can't otherwise drive). Registered
        # by the orchestrator via ``set_fixtures`` before a query runs. A
        # context-level ``filechooser`` listener intercepts the dialog and
        # auto-attaches ``_active_fixtures`` (defaults to all registered
        # files), so this works for BOTH the L-DOM and L-PIX tracks without
        # the agent needing DOM access. The ``upload_file`` agent tool just
        # selects WHICH fixture to attach, then clicks the upload control.
        self._fixtures: dict[str, str] = {}        # name -> absolute path
        self._fixture_order: list[str] = []        # registration order (names)
        self._active_fixtures: list[str] = []      # absolute paths to attach next

    async def start(self) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        self._context = await self._new_configured_context()
        self._page = await self._new_configured_page(self._context)

    async def _new_configured_context(self) -> BrowserContext:
        """Create a BrowserContext with the shared preview/clock plumbing.

        Factored out of :meth:`start` so ``spawn_isolated_session`` can build a
        second, fully-isolated context (fresh cookies / localStorage / anonymous
        identity) with the exact same Cloudflare-bypass header scoping and
        virtual-clock shim. Keeping one code path guarantees a viewer subagent
        clears the preview challenge the same way the main session does.
        """
        if self._browser is None:
            raise RuntimeError("Browser not started. Call start() first.")
        context = await self._browser.new_context(viewport=self._viewport)
        # Inject the preview-access header ONLY for same-origin requests to the
        # preview host. Using context-level ``extra_http_headers`` would add it
        # to EVERY request (incl. cross-origin map/font CDNs), where the custom
        # header forces a CORS preflight those hosts reject -> map never loads.
        if self._preview_secret:
            logger.info(
                "PREVIEW_TEST_SECRET set; scoping X-Preview-Test header to "
                "same-origin preview requests only"
            )
            await context.route("**/*", self._inject_preview_header)
        # Register the virtual-clock shim on the CONTEXT so it auto-applies
        # to this page and any page created later via ``new_page``. Must run
        # before the first ``new_page`` so the very first document gets it.
        if self._enable_virtual_clock:
            try:
                await context.add_init_script(_VIRTUAL_CLOCK_SHIM)
            except Exception as e:  # noqa: BLE001
                logger.warning("failed to inject virtual clock shim: %s", e)
        # TTS shim is always injected: it is inert unless the app calls
        # speechSynthesis.speak(), so normal apps are unaffected.
        try:
            await context.add_init_script(_TTS_SHIM)
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to inject TTS shim: %s", e)
        return context

    async def _new_configured_page(self, context: BrowserContext) -> Page:
        """Open a page in ``context`` with the standard timeout + listeners."""
        page = await context.new_page()
        page.set_default_timeout(30_000)
        self._attach_runtime_listeners(page)
        self._attach_filechooser_listener(page)
        return page

    async def stop(self) -> None:
        # Close any viewer subagent sessions still open (best-effort) before
        # tearing down the shared Browser they borrow from us.
        for child in list(self._child_sessions):
            try:
                await self.close_session(child)
            except Exception as e:  # noqa: BLE001
                logger.warning("failed to close child session on stop: %s", e)
        if self._context:
            await self._context.close()
        # Child envs share the parent Browser/Playwright and must not close
        # them; only the owner tears the stack down.
        if self._browser and self._owns_browser:
            await self._browser.close()
        if self._playwright and self._owns_browser:
            await self._playwright.stop()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    # ── Multi-session: isolated viewer sub-sessions (0702) ──────────

    async def spawn_isolated_session(self, url: str | None = None) -> "PlaywrightEnv":
        """Open a fresh, isolated session in the SAME browser and return it.

        Used by the ``delegate_to_viewer`` tool so the main CUA can hand a
        share link to a second "person" (viewer B). The returned object is a
        fully-usable :class:`PlaywrightEnv` (own screenshots / mouse / keyboard
        / runtime signals) backed by a brand-new ``BrowserContext`` — i.e. it
        carries NONE of the main session's cookies / localStorage / identity,
        which is exactly what realtime-sync and per-identity-isolation checks
        require. It reuses the parent's preview header + virtual clock so it
        clears Cloudflare on the same origin.

        The child does not own the Browser/Playwright, so its ``stop`` (or the
        preferred :meth:`close_session`) only drops its own context.
        """
        if self._browser is None:
            raise RuntimeError("Browser not started. Call start() first.")
        child = PlaywrightEnv(
            viewport=dict(self._viewport),
            headless=self._headless,
            enable_virtual_clock=self._enable_virtual_clock,
            fixture_roots=list(self._fixture_roots),
        )
        # Wire the child onto the shared Browser without a Playwright of its
        # own, and carry over the preview-secret scoping so its header route
        # matches the same-origin preview host.
        child._playwright = None
        child._browser = self._browser
        child._owns_browser = False
        child._preview_secret = self._preview_secret
        child._context = await child._new_configured_context()
        child._page = await child._new_configured_page(child._context)
        self._child_sessions.append(child)
        if url:
            ok = await child.goto(url, retries=1)
            if not ok:
                logger.warning("viewer session goto(%s) did not fully succeed", url)
        return child

    async def close_session(self, session: "PlaywrightEnv") -> None:
        """Close a viewer sub-session's context (its "logout").

        For an in-app anonymous share link there is no server account to sign
        out of; discarding the context throws away its cookies / localStorage /
        anonymous state cleanly and leaves the main session untouched.
        """
        if session is self:
            raise ValueError("refusing to close the main session via close_session")
        try:
            if session._context is not None:
                await session._context.close()
        finally:
            session._page = None
            session._context = None
            if session in self._child_sessions:
                self._child_sessions.remove(session)

    async def _inject_preview_header(self, route: Any) -> None:
        """Attach ``X-Preview-Test`` only to same-origin preview requests.

        The header is required to clear the preview's Cloudflare challenge,
        but it must NOT leak onto cross-origin requests (map tiles, fonts):
        a non-standard header turns those into CORS-preflighted requests that
        the third-party CDNs reject, which silently blocks the map. We match
        the navigated preview host (captured in :meth:`goto`); same-origin
        requests never trigger a preflight, so the header is safe there.
        """
        try:
            req = route.request
            host = (urlparse(req.url).hostname or "").lower()
            if self._preview_secret and host and host == self._preview_host:
                headers = dict(req.headers)
                headers["x-preview-test"] = self._preview_secret
                await route.continue_(headers=headers)
            else:
                await route.continue_()
        except Exception as e:  # noqa: BLE001
            logger.debug("preview header route failed: %s", e)
            try:
                await route.continue_()
            except Exception:  # noqa: BLE001
                pass

    async def new_page(self) -> Page:
        """Create a fresh page within the same browser context."""
        if self._context is None:
            raise RuntimeError("Browser not started. Call start() first.")
        self._page = await self._context.new_page()
        self._page.set_default_timeout(30_000)
        self._attach_runtime_listeners(self._page)
        self._attach_filechooser_listener(self._page)
        return self._page

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("No active page. Call start() first.")
        return self._page

    # ── File-upload fixtures ────────────────────────────────────────

    def set_fixtures(
        self,
        fixtures: list[dict[str, Any]] | list[str] | None,
        base_dir: str | Path | None = None,
    ) -> list[str]:
        """Register the file-upload fixtures available for the current query.

        ``fixtures`` accepts either a list of ``{"name": ..., "path": ...}``
        dicts or a bare list of path strings (name defaults to the filename).
        Relative paths are resolved against ``base_dir`` (default: CWD).
        Missing files are skipped with a warning. Resets the active selection
        to ALL registered fixtures so a transparent auto-attach works even if
        the agent never calls ``upload_file`` explicitly.

        Returns the list of fixture NAMES successfully registered.
        """
        self._fixtures = {}
        self._fixture_order = []
        base = Path(base_dir) if base_dir is not None else Path.cwd()
        for item in fixtures or []:
            if isinstance(item, str):
                name, raw_path = "", item
            elif isinstance(item, dict):
                name, raw_path = str(item.get("name") or ""), str(item.get("path") or "")
            else:
                continue
            if not raw_path:
                continue
            p = Path(raw_path)
            if not p.is_absolute():
                # Resolution order:
                #   1. FIXTURES_DIR env roots (':'-separated) — runtime
                #      override without touching config;
                #   2. configured roots (``fixtures_dirs`` in the eval YAML);
                #   3. the explicit/CWD base and the repo root, so repo-relative
                #      paths like ``data/mock_resources/x.png`` keep working;
                #   4. DEFAULT_FIXTURE_ROOTS — the known CN/EN 0710 asset
                #      folders whose seeds use paths like
                #      "S1-011/cat_a_avatar.png".
                env_roots = [
                    Path(r.strip())
                    for r in os.environ.get("FIXTURES_DIR", "").split(os.pathsep)
                    if r.strip()
                ]
                candidates = (
                    [root / p for root in env_roots]
                    + [root / p for root in self._fixture_roots]
                    + [base / p, REPO_DIR / p]
                    + [root / p for root in DEFAULT_FIXTURE_ROOTS]
                )
                resolved = next((c for c in candidates if c.exists()), None)
                if resolved is None:
                    logger.warning(
                        "fixture not found (tried %s), skipping: %s",
                        ", ".join(str(c) for c in candidates), raw_path,
                    )
                    continue
                p = resolved
            elif not p.exists():
                logger.warning("fixture not found, skipping: %s", p)
                continue
            if not name:
                name = p.name
            self._fixtures[name] = str(p)
            self._fixture_order.append(name)
        self._active_fixtures = [self._fixtures[n] for n in self._fixture_order]
        if self._fixture_order:
            logger.info("registered %d upload fixture(s): %s",
                        len(self._fixture_order), self._fixture_order)
        return list(self._fixture_order)

    def available_fixtures(self) -> list[str]:
        """Names of the fixtures registered for the current query."""
        return list(self._fixture_order)

    def select_fixtures(self, names: list[str] | str | None) -> list[str]:
        """Choose WHICH registered fixtures the next file-picker receives.

        ``None`` selects all registered fixtures. Unknown names are ignored.
        Returns the resolved absolute paths now active.
        """
        if names is None:
            self._active_fixtures = [self._fixtures[n] for n in self._fixture_order]
            return list(self._active_fixtures)
        if isinstance(names, str):
            names = [names]
        paths: list[str] = []
        for n in names:
            if n in self._fixtures:
                paths.append(self._fixtures[n])
            else:
                # Allow selecting by filename even if registered under a name.
                for reg_name, reg_path in self._fixtures.items():
                    if Path(reg_path).name == n:
                        paths.append(reg_path)
                        break
        self._active_fixtures = paths
        return list(self._active_fixtures)

    def _attach_filechooser_listener(self, page: Page) -> None:
        """Intercept native file pickers and auto-attach the active fixtures.

        Registered on every page. When no fixtures are active the chooser is
        left untouched (Playwright still suppresses the blocking OS dialog).
        """
        def _on_filechooser(chooser: Any) -> None:
            files = list(self._active_fixtures)
            if not files:
                logger.info("filechooser opened but no fixtures active; ignoring")
                return
            try:
                payload = files if chooser.is_multiple() else files[0]
                asyncio.ensure_future(chooser.set_files(payload))
                logger.info("filechooser auto-attached fixtures: %s", files)
            except Exception as e:  # noqa: BLE001
                logger.warning("failed to attach fixtures to filechooser: %s", e)

        try:
            page.on("filechooser", _on_filechooser)
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to register filechooser listener: %s", e)

    async def goto(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
        *,
        retries: int = 0,
        retry_backoff_ms: int = 500,
    ) -> bool:
        """Navigate to ``url`` with optional retry on transient network errors.

        ``domcontentloaded`` is much faster than ``networkidle`` on SPAs that
        keep long-poll / websocket connections open. The CUA loop already
        waits ~800ms before the first action and takes a fresh screenshot
        each step, so we don't need to wait for full network idleness here.

        Returns ``True`` if the navigation eventually succeeded (including
        the ``commit`` fallback), ``False`` if every attempt failed. Callers
        can combine the return value with :meth:`get_page_info` to decide
        whether the resulting page is actually usable.
        """
        # Remember the preview origin so the header-injection route knows which
        # requests are same-origin (and therefore safe to tag with the secret).
        if self._preview_secret:
            self._preview_host = (urlparse(url).hostname or "").lower()
        last_err: Exception | None = None
        for attempt in range(max(1, retries + 1)):
            try:
                await self.page.goto(url, wait_until=wait_until, timeout=15_000)
                return True
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning(
                    "goto %s failed (attempt %d/%d, wait_until=%s): %s",
                    url, attempt + 1, retries + 1, wait_until, e,
                )
                if attempt < retries:
                    await asyncio.sleep(retry_backoff_ms / 1000.0 * (attempt + 1))

        try:
            await self.page.goto(url, wait_until="commit", timeout=10_000)
            logger.warning(
                "goto %s recovered via wait_until=commit after %d failures",
                url, retries + 1,
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(
                "goto %s failed completely after %d retries (last_err=%s, "
                "commit_err=%s)",
                url, retries + 1, last_err, e,
            )
            return False

    async def reload(
        self,
        wait_until: str = "domcontentloaded",
        *,
        retries: int = 0,
        retry_backoff_ms: int = 500,
    ) -> bool:
        """Reload the current page with retry semantics matching :meth:`goto`.

        Returns ``True`` on success, ``False`` if every attempt fails. Used
        both as a CUA tool (so the model can self-recover from a blank /
        half-loaded page) and as the orchestrator's last-mile retry before
        triggering the J1 ``page_load_failure`` cascade.
        """
        last_err: Exception | None = None
        for attempt in range(max(1, retries + 1)):
            try:
                await self.page.reload(wait_until=wait_until, timeout=15_000)
                return True
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning(
                    "reload failed (attempt %d/%d, wait_until=%s): %s",
                    attempt + 1, retries + 1, wait_until, e,
                )
                if attempt < retries:
                    await asyncio.sleep(retry_backoff_ms / 1000.0 * (attempt + 1))

        try:
            await self.page.reload(wait_until="commit", timeout=10_000)
            logger.warning(
                "reload recovered via wait_until=commit after %d failures",
                retries + 1,
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(
                "reload failed completely after %d retries (last_err=%s, "
                "commit_err=%s)",
                retries + 1, last_err, e,
            )
            return False

    async def screenshot_bytes(self) -> bytes:
        return await self.page.screenshot(full_page=False)

    async def screenshot_bytes_stable(
        self,
        *,
        max_wait_ms: int = 800,
        settle_ms: int = 120,
        wait_load: bool = False,
    ) -> bytes:
        """Screenshot only AFTER the viewport has visually settled.

        Fixes the "click -> screenshot caught mid-animation/render" failure
        mode (recommendation cards mid-transition, pie chart still drawing).
        Strategy: take consecutive frames ``settle_ms`` apart and return as
        soon as two are byte-identical (no animation in flight), or return
        the last frame once ``max_wait_ms`` is exhausted (bounded cost).

        ``wait_load`` (L-DOM track) additionally waits briefly for network
        idle first; failures are swallowed so a long-poll SPA never hangs us.
        """
        import hashlib

        if wait_load:
            try:
                await self.page.wait_for_load_state("networkidle", timeout=1000)
            except Exception:  # noqa: BLE001
                pass

        max_wait_ms = max(0, int(max_wait_ms))
        settle_ms = max(20, int(settle_ms))
        prev = await self.page.screenshot(full_page=False)
        prev_h = hashlib.sha1(prev).digest()
        elapsed = 0
        while elapsed < max_wait_ms:
            await asyncio.sleep(settle_ms / 1000.0)
            elapsed += settle_ms
            cur = await self.page.screenshot(full_page=False)
            cur_h = hashlib.sha1(cur).digest()
            if cur_h == prev_h:
                return cur
            prev, prev_h = cur, cur_h
        return prev

    async def screenshot_base64(self) -> str:
        raw = await self.screenshot_bytes()
        return base64.b64encode(raw).decode("utf-8")

    async def save_screenshot(self, path: str | Path) -> str:
        """Take screenshot and save to file. Returns the path as string."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = await self.screenshot_bytes()
        p.write_bytes(raw)
        return str(p)

    async def get_page_info(self) -> dict[str, Any]:
        """Collect current page state as a dict."""
        title = await self.page.title()
        url = self.page.url
        raw_text = await self.page.evaluate(
            "() => document.body ? document.body.innerText : ''"
        )
        if not isinstance(raw_text, str):
            raw_text = ""
        return {
            "page_url": url,
            "page_title": title,
            "visible_text": raw_text[:VISIBLE_TEXT_MAX_CHARS],
            "visible_text_raw_len": len(raw_text),
        }

    async def get_viewport_size(self) -> dict[str, int]:
        return dict(self._viewport)

    # ── Mouse primitives (support button & modifiers) ──────────────

    _PLAYWRIGHT_MODIFIER_MAP = {
        "shift": "Shift",
        "ctrl": "Control",
        "control": "Control",
        "alt": "Alt",
        "meta": "Meta",
        "cmd": "Meta",
        "super": "Meta",
    }

    @classmethod
    def _normalize_modifiers(cls, modifiers: list[str] | None) -> list[str]:
        if not modifiers:
            return []
        return [cls._PLAYWRIGHT_MODIFIER_MAP.get(m.lower(), m) for m in modifiers]

    async def click_at(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
        modifiers: list[str] | None = None,
    ) -> None:
        # Playwright Python's Mouse.click() does NOT accept a `modifiers` kwarg
        # (only Page.click() does). Emulate it by holding the modifier keys
        # around the mouse click.
        norm = self._normalize_modifiers(modifiers)
        for m in norm:
            await self.page.keyboard.down(m)
        try:
            await self.page.mouse.click(
                x, y,
                button=button,
                click_count=clicks,
            )
            self._last_cursor_xy = (int(x), int(y))
        finally:
            for m in reversed(norm):
                await self.page.keyboard.up(m)

    async def double_click_at(self, x: int, y: int) -> None:
        await self.page.mouse.dblclick(x, y)
        self._last_cursor_xy = (int(x), int(y))

    async def mouse_down_at(
        self, x: int, y: int, button: str = "left",
    ) -> None:
        await self.page.mouse.move(x, y)
        await self.page.mouse.down(button=button)
        self._last_cursor_xy = (int(x), int(y))

    async def mouse_up_at(
        self, x: int, y: int, button: str = "left",
    ) -> None:
        await self.page.mouse.move(x, y)
        await self.page.mouse.up(button=button)
        self._last_cursor_xy = (int(x), int(y))

    async def type_text(self, text: str) -> None:
        await self.page.keyboard.type(text, delay=30)

    async def press_key(self, key: str) -> None:
        await self.page.keyboard.press(key)

    async def hold_key(self, key: str, duration_ms: int) -> None:
        """Press the key, hold for duration_ms, then release."""
        await self.page.keyboard.down(key)
        try:
            await asyncio.sleep(max(0.0, duration_ms / 1000.0))
        finally:
            await self.page.keyboard.up(key)

    async def scroll(
        self,
        x: int,
        y: int,
        delta_x: int = 0,
        delta_y: int = 0,
        modifiers: list[str] | None = None,
    ) -> None:
        await self.page.mouse.move(x, y)
        self._last_cursor_xy = (int(x), int(y))
        norm = self._normalize_modifiers(modifiers)
        for m in norm:
            await self.page.keyboard.down(m)
        try:
            await self.page.mouse.wheel(delta_x, delta_y)
        finally:
            for m in reversed(norm):
                await self.page.keyboard.up(m)

    async def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        button: str = "left",
    ) -> None:
        await self.page.mouse.move(start_x, start_y)
        await self.page.mouse.down(button=button)
        await self.page.mouse.move(end_x, end_y, steps=10)
        await self.page.mouse.up(button=button)
        self._last_cursor_xy = (int(end_x), int(end_y))

    async def move_to(self, x: int, y: int) -> None:
        await self.page.mouse.move(x, y)
        self._last_cursor_xy = (int(x), int(y))

    def get_cursor_position(self) -> dict[str, int]:
        """Return the last known cursor position as ``{"x": int, "y": int}``.

        Updated by every mouse primitive in this class (``click_at`` /
        ``move_to`` / ``drag`` / ``scroll`` / ``mouse_down_at`` /
        ``mouse_up_at``). The Anthropic native ``cursor_position`` action
        consumes this value via the ``__cursor__`` sentinel in
        ``native_adapter.py``. We keep it as a cached field instead of a
        live Playwright query because Playwright does not expose a mouse-
        position read on ``page.mouse`` and asking the DOM for it would
        cost an extra ``evaluate()`` round-trip per action.
        """
        x, y = self._last_cursor_xy
        return {"x": int(x), "y": int(y)}

    async def wait(self, ms: int = 500) -> None:
        await asyncio.sleep(ms / 1000.0)

    # ── Virtual clock control ──────────────────────────────────────

    async def advance_clock(self, ms: int) -> dict[str, Any]:
        """Fast-forward the in-page virtual clock by ``ms`` milliseconds.

        Synchronously fires every timer (setTimeout/setInterval/rAF) whose
        virtual deadline falls inside the advanced window and shifts the
        value returned by ``Date.now`` / ``new Date()`` / ``performance.now``.
        After firing we yield a short *real* sleep so the framework (React,
        etc.) can flush state updates and paint the resulting frame before
        the next screenshot is taken.

        Returns ``{"ok": bool, "offset_ms": int, "pending_timers": int,
        "error": str|None}``. Never raises — a disabled clock or a page
        that hasn't installed the shim yet returns ``ok=False`` so callers
        can degrade gracefully.
        """
        if not self._enable_virtual_clock:
            return {"ok": False, "error": "virtual_clock_disabled", "offset_ms": 0}
        if self._page is None:
            return {"ok": False, "error": "no_active_page", "offset_ms": 0}
        clamped = max(0, min(int(ms or 0), MAX_ADVANCE_MS))
        try:
            res = await self._page.evaluate(
                "(ms) => (window.__advanceVirtualClock "
                "? window.__advanceVirtualClock(ms) : null)",
                clamped,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("advance_clock failed: %s", e)
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "offset_ms": 0}
        if not isinstance(res, dict):
            return {
                "ok": False,
                "error": "virtual_clock_not_installed",
                "offset_ms": 0,
            }
        # Let the app re-render / paint after the burst of fired timers.
        await self.wait(400)
        return {
            "ok": True,
            "error": None,
            "advanced_ms": clamped,
            "offset_ms": int(res.get("offset_ms", 0) or 0),
            "pending_timers": int(res.get("pending_timers", 0) or 0),
        }

    async def reset_clock(self) -> dict[str, Any]:
        """Reset the virtual-clock offset back to 0 (real time)."""
        if not self._enable_virtual_clock or self._page is None:
            return {"ok": False, "offset_ms": 0}
        try:
            res = await self._page.evaluate(
                "() => (window.__resetVirtualClock "
                "? window.__resetVirtualClock() : null)"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("reset_clock failed: %s", e)
            return {"ok": False, "offset_ms": 0}
        return {"ok": isinstance(res, dict), "offset_ms": 0}

    # ── Image helpers ──────────────────────────────────────────────

    async def screenshot_region_base64(
        self, x1: int, y1: int, x2: int, y2: int,
    ) -> str:
        """Return a base64 PNG cropped to the given region (clamped to viewport)."""
        vp = self._viewport
        max_x = vp.get("width", 1280)
        max_y = vp.get("height", 900)
        lx = max(0, min(int(x1), max_x - 1))
        ly = max(0, min(int(y1), max_y - 1))
        rx = max(lx + 1, min(int(x2), max_x))
        ry = max(ly + 1, min(int(y2), max_y))
        clip = {"x": lx, "y": ly, "width": rx - lx, "height": ry - ly}
        raw = await self.page.screenshot(full_page=False, clip=clip)
        return base64.b64encode(raw).decode("utf-8")

    # ── Accessibility ──────────────────────────────────────────────

    async def accessibility_snapshot(self, interesting_only: bool = True) -> dict[str, Any] | None:
        """Return the Chromium accessibility tree as a dict, or None on failure."""
        try:
            return await self.page.accessibility.snapshot(interesting_only=interesting_only)
        except Exception as e:
            logger.warning("accessibility snapshot failed: %s", e)
            return None

    # ── Runtime signals (console / network / hit-test) ────────────

    def _attach_runtime_listeners(self, page: Page) -> None:
        """Wire console / pageerror / requestfailed / response listeners.

        We deliberately swallow any exception inside the listeners so a
        broken site cannot kill the verification pipeline.
        """

        def _on_console(msg: Any) -> None:
            try:
                if msg.type != "error":
                    return
                if len(self._console_errors) >= MAX_CONSOLE_ERRORS:
                    return
                loc = getattr(msg, "location", None) or {}
                self._console_errors.append({
                    "type": "console.error",
                    "text": _truncate(str(msg.text or ""), 600),
                    "url": (loc or {}).get("url") if isinstance(loc, dict) else None,
                    "line_no": (loc or {}).get("lineNumber") if isinstance(loc, dict) else None,
                    "ts_ms": _now_ms(),
                })
            except Exception as e:  # noqa: BLE001
                logger.debug("console listener failed: %s", e)

        def _on_pageerror(err: Any) -> None:
            try:
                if len(self._console_errors) >= MAX_CONSOLE_ERRORS:
                    return
                self._console_errors.append({
                    "type": "pageerror",
                    "text": _truncate(str(err), 600),
                    "url": None,
                    "line_no": None,
                    "ts_ms": _now_ms(),
                })
            except Exception as e:  # noqa: BLE001
                logger.debug("pageerror listener failed: %s", e)

        def _on_requestfailed(req: Any) -> None:
            try:
                self._record_external_host(getattr(req, "url", "") or "")
                if len(self._failed_requests) >= MAX_FAILED_REQUESTS:
                    return
                fail = getattr(req, "failure", None)
                fail_text = ""
                try:
                    fail_text = (fail or {}).get("errorText", "") if isinstance(fail, dict) else str(fail or "")
                except Exception:
                    fail_text = ""
                url = getattr(req, "url", "") or ""
                net_trace.browser_event(
                    "requestfailed",
                    method=getattr(req, "method", "") or "",
                    url=_truncate(url, 400),
                    error=_truncate(fail_text, 200),
                )
                self._failed_requests.append({
                    "kind": "requestfailed",
                    "method": getattr(req, "method", "") or "",
                    "url": _truncate(url, 400),
                    "host": _host_of(url),
                    "status": None,
                    "error": _truncate(fail_text, 200),
                    "ts_ms": _now_ms(),
                })
            except Exception as e:  # noqa: BLE001
                logger.debug("requestfailed listener failed: %s", e)

        def _on_response(resp: Any) -> None:
            try:
                url = getattr(resp, "url", "") or ""
                self._record_external_host(url)
                status = int(getattr(resp, "status", 0) or 0)
                req = getattr(resp, "request", None)
                method = getattr(req, "method", "") if req is not None else ""

                if net_trace.enabled() and not url.startswith(("data:", "blob:")):
                    net_trace.browser_event(
                        "response",
                        method=method,
                        url=_truncate(url, 400),
                        status=status,
                    )

                # Runtime audit channel: record every backend/API call,
                # success or failure, so meta-review can distinguish
                # "feature exercised, backend OK" from "backend rejected".
                if self._is_backend_url(url):
                    self._record_backend_call(resp, url, method, status)

                if status < 400:
                    return
                if len(self._failed_requests) >= MAX_FAILED_REQUESTS:
                    return
                self._failed_requests.append({
                    "kind": f"response_{status // 100}xx",
                    "method": method,
                    "url": _truncate(url, 400),
                    "host": _host_of(url),
                    "status": status,
                    "error": "",
                    "ts_ms": _now_ms(),
                })
            except Exception as e:  # noqa: BLE001
                logger.debug("response listener failed: %s", e)

        try:
            page.on("console", _on_console)
            page.on("pageerror", _on_pageerror)
            page.on("requestfailed", _on_requestfailed)
            page.on("response", _on_response)
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to attach runtime listeners: %s", e)

    def _record_external_host(self, url: str) -> None:
        host = _host_of(url)
        if not host:
            return
        if host in _DEV_HOSTS_BLOCKLIST:
            return
        page_host = _host_of(self._page.url) if self._page is not None else ""
        is_external = bool(page_host) and host != page_host
        entry = self._external_hosts.get(host)
        if entry is None:
            if len(self._external_hosts) >= MAX_EXTERNAL_HOSTS:
                return
            self._external_hosts[host] = {
                "host": host,
                "count": 1,
                "sample_url": _truncate(url, 240),
                "is_external": is_external,
            }
        else:
            entry["count"] = int(entry.get("count", 0)) + 1
            entry["is_external"] = entry.get("is_external", False) or is_external

    def _is_backend_url(self, url: str) -> bool:
        """True when ``url`` looks like a backend/API call worth auditing.

        Two ways to qualify: (a) cross-origin to the app (different host,
        not a dev-loopback host), or (b) the path matches a known API
        marker (Supabase REST/auth/storage/functions, ``/rpc/``, ``/api/``)
        even when same-origin via a proxy. Static asset hosts that merely
        differ from the page host (CDNs for fonts/images) still get
        recorded; that's acceptable noise for v1 and is easy to filter
        downstream by status/method.
        """
        if not url:
            return False
        lowered = url.lower()
        if any(marker in lowered for marker in _API_PATH_MARKERS):
            return True
        host = _host_of(url)
        if not host or host in _DEV_HOSTS_BLOCKLIST:
            return False
        page_host = _host_of(self._page.url) if self._page is not None else ""
        return bool(page_host) and host != page_host

    def _record_backend_call(
        self, resp: Any, url: str, method: str, status: int,
    ) -> None:
        """Append a backend call to the audit channel; fetch error body async.

        The entry is appended synchronously so ordering is preserved; for
        4xx/5xx we schedule an async task to read the response body and
        backfill ``pg_code`` / ``pg_message`` / ``body_excerpt`` onto the
        same dict. The task is best-effort: any failure leaves the entry
        as-is (status is still recorded).
        """
        if len(self._backend_calls) >= MAX_BACKEND_CALLS:
            return
        entry: dict[str, Any] = {
            "method": method,
            "url": _truncate(url, 400),
            "host": _host_of(url),
            "status": status,
            "ok": status < 400,
            "pg_code": None,
            "pg_message": None,
            "body_excerpt": None,
            "ts_ms": _now_ms(),
        }
        self._backend_calls.append(entry)
        if status >= 400:
            try:
                asyncio.get_running_loop().create_task(
                    self._fill_backend_error_body(resp, entry)
                )
            except RuntimeError:
                # No running loop (should not happen under async_api); skip
                # the body fetch but keep the status-only entry.
                pass

    async def _fill_backend_error_body(
        self, resp: Any, entry: dict[str, Any],
    ) -> None:
        """Read a 4xx/5xx response body and extract PostgreSQL/PostgREST codes.

        Supabase (PostgREST) error payloads look like
        ``{"code":"42501","message":"...","details":...,"hint":...}`` or
        ``{"message":"...","code":"PGRST116"}``. We surface ``code`` as
        ``pg_code`` and ``message`` as ``pg_message`` so the deterministic
        join in meta-review / scoring has a stable root-cause key.
        """
        try:
            text = await resp.text()
        except Exception as e:  # noqa: BLE001
            logger.debug("backend error body read failed: %s", e)
            return
        if not isinstance(text, str) or not text:
            return
        entry["body_excerpt"] = _truncate(text, BACKEND_BODY_MAX_CHARS)
        try:
            import json as _json
            data = _json.loads(text)
        except Exception:
            return
        if isinstance(data, dict):
            code = data.get("code")
            msg = data.get("message")
            if code is not None:
                entry["pg_code"] = str(code)
            if msg is not None:
                entry["pg_message"] = _truncate(str(msg), 300)

    async def drain_tts_events(self) -> list[dict[str, Any]]:
        """Sync ``window.__ttsLog`` into the Python buffer; return NEW entries.

        Called after every CUA action so the runner can surface fresh TTS
        events in the action feedback text. Entries already ingested are
        updated in-place (the page mutates ``started``/``ended``/``error``
        asynchronously as the utterance plays), so the snapshot taken at
        claim end carries the final lifecycle flags. Never raises.
        """
        if self._page is None:
            return []
        try:
            res = await self._page.evaluate(
                "() => (window.__ttsLog "
                "? {gen: window.__ttsLogGen || '', events: window.__ttsLog} "
                ": null)"
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("drain_tts_events failed: %s", e)
            return []
        if not isinstance(res, dict):
            return []
        gen = str(res.get("gen") or "")
        events = res.get("events")
        if not isinstance(events, list):
            return []
        new_entries: list[dict[str, Any]] = []
        for idx, ev in enumerate(events):
            if not isinstance(ev, dict):
                continue
            key = (gen, idx)
            existing = self._tts_index.get(key)
            if existing is not None:
                # Merge late lifecycle updates onto the buffered entry.
                for field_name in ("started", "ended", "error"):
                    existing[field_name] = ev.get(field_name)
                continue
            if len(self._tts_events) >= MAX_TTS_EVENTS:
                continue
            entry = {
                "event": str(ev.get("event") or "speak"),
                "text": _truncate(str(ev.get("text") or ""), 300),
                "lang": str(ev.get("lang") or ""),
                "voice": str(ev.get("voice") or ""),
                "started": ev.get("started"),
                "ended": ev.get("ended"),
                "error": ev.get("error"),
                "ts_ms": ev.get("ts"),
            }
            self._tts_events.append(entry)
            self._tts_index[key] = entry
            new_entries.append(entry)
        return new_entries

    def reset_runtime_log(self) -> None:
        """Clear runtime signals. Call before each new claim verification."""
        self._console_errors.clear()
        self._failed_requests.clear()
        self._backend_calls.clear()
        self._external_hosts.clear()
        self._hit_tests.clear()
        self._tts_events.clear()
        self._tts_index.clear()
        self._runtime_started_at_ms = _now_ms()

    def runtime_signals_snapshot(self) -> dict[str, Any]:
        """Return the runtime signals collected since the last reset."""
        return {
            "console_errors": list(self._console_errors),
            "failed_requests": list(self._failed_requests),
            "backend_calls": list(self._backend_calls),
            "external_hosts_called": [
                dict(v) for v in self._external_hosts.values()
            ],
            "hit_test_results": list(self._hit_tests),
            "tts_events": [dict(e) for e in self._tts_events],
        }

    async def extract_map_geometry(self) -> dict[str, Any]:
        """Best-effort extraction of map geometry from the live page.

        Returns a dict with:
          - ``polylines``: list of ``{vertex_count, screen_points,
            bbox: {x_min,x_max,y_min,y_max}, span_x, span_y}``
          - ``markers``:   list of ``{x, y, label}``
          - ``map_containers``: how many ``.leaflet-container`` / ``.mapboxgl-map``
            / ``[class*="amap"]`` containers were found
          - ``library``: best guess of the map library

        Used by ``programmatic_validators.validate_geo_semantics`` to detect
        cases where a route is drawn as a single straight line (only 2
        vertices) instead of a polyline that follows real lat/lng. We
        deliberately do NOT raise — on failure we return empty lists.
        """
        try:
            return await self.page.evaluate(
                """
                () => {
                  const out = {
                    polylines: [],
                    markers: [],
                    map_containers: 0,
                    library: 'unknown',
                  };
                  const containers = document.querySelectorAll(
                    '.leaflet-container, .mapboxgl-map, .amap-container, '
                    + '[class*="MapContainer"], [class*="map-container"]'
                  );
                  out.map_containers = containers.length;
                  if (document.querySelector('.leaflet-container')) out.library = 'leaflet';
                  else if (document.querySelector('.mapboxgl-map')) out.library = 'mapbox-gl';
                  else if (document.querySelector('.amap-container')) out.library = 'amap';
                  else if (window.L && window.L.version) out.library = 'leaflet';
                  else if (window.AMap) out.library = 'amap';
                  else if (window.mapboxgl) out.library = 'mapbox-gl';

                  const svgPaths = document.querySelectorAll(
                    '.leaflet-overlay-pane svg path, svg path[d*="M"]'
                  );
                  for (const p of svgPaths) {
                    const d = p.getAttribute('d') || '';
                    if (!d) continue;
                    const re = /(-?\\d+(?:\\.\\d+)?)[ ,]+(-?\\d+(?:\\.\\d+)?)/g;
                    const pts = [];
                    let m;
                    while ((m = re.exec(d)) !== null) {
                      const x = parseFloat(m[1]);
                      const y = parseFloat(m[2]);
                      if (Number.isFinite(x) && Number.isFinite(y)) pts.push({x, y});
                    }
                    if (pts.length < 2) continue;
                    let xMin = pts[0].x, xMax = pts[0].x;
                    let yMin = pts[0].y, yMax = pts[0].y;
                    for (const pt of pts) {
                      if (pt.x < xMin) xMin = pt.x;
                      if (pt.x > xMax) xMax = pt.x;
                      if (pt.y < yMin) yMin = pt.y;
                      if (pt.y > yMax) yMax = pt.y;
                    }
                    const head = pts.slice(0, 8);
                    const tail = pts.length > 8 ? pts.slice(-4) : [];
                    out.polylines.push({
                      vertex_count: pts.length,
                      screen_points: head.concat(tail),
                      bbox: {x_min: xMin, x_max: xMax, y_min: yMin, y_max: yMax},
                      span_x: xMax - xMin,
                      span_y: yMax - yMin,
                    });
                  }

                  const markerSel = (
                    '.leaflet-marker-icon, '
                    + '.mapboxgl-marker, '
                    + '.amap-marker, '
                    + '[class*="map-marker"], '
                    + '[class*="MapMarker"], '
                    + '[data-testid*="marker"]'
                  );
                  for (const el of document.querySelectorAll(markerSel)) {
                    const r = el.getBoundingClientRect();
                    if (r.width === 0 && r.height === 0) continue;
                    out.markers.push({
                      x: r.x + r.width / 2,
                      y: r.y + r.height / 2,
                      label: (el.getAttribute('aria-label') || el.title || '').slice(0, 80),
                    });
                  }
                  return out;
                }
                """
            ) or {}
        except Exception as e:  # noqa: BLE001
            logger.warning("extract_map_geometry failed: %s", e)
            return {
                "polylines": [],
                "markers": [],
                "map_containers": 0,
                "library": "unknown",
                "_error": f"{type(e).__name__}: {e}",
            }

    async def hit_test(
        self,
        x: int,
        y: int,
        *,
        expected_text: str | None = None,
        expected_role: str | None = None,
        label: str = "",
    ) -> dict[str, Any]:
        """Run ``document.elementFromPoint(x, y)`` and report what's on top.

        Used by ``interaction_accessibility`` claims to detect cases where a
        button is rendered but covered by another layer / has
        ``pointer-events:none``.
        """
        try:
            info = await self.page.evaluate(
                """
                ({x, y}) => {
                  const el = document.elementFromPoint(x, y);
                  if (!el) return null;
                  const cs = window.getComputedStyle(el);
                  const rect = el.getBoundingClientRect();
                  const text = (el.innerText || el.textContent || "").trim().slice(0, 120);
                  return {
                    tag: el.tagName.toLowerCase(),
                    role: el.getAttribute('role'),
                    aria_label: el.getAttribute('aria-label'),
                    id: el.id || null,
                    cls: el.className && typeof el.className === 'string'
                      ? el.className.slice(0, 200) : null,
                    text,
                    pointer_events: cs.pointerEvents,
                    visibility: cs.visibility,
                    display: cs.display,
                    z_index: cs.zIndex,
                    rect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height },
                  };
                }
                """,
                {"x": int(x), "y": int(y)},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("hit_test evaluate failed: %s", e)
            info = None

        hit_match = False
        if info:
            text = (info.get("text") or "").lower()
            role = (info.get("role") or "").lower()
            aria = (info.get("aria_label") or "").lower()
            if expected_text and expected_text.lower() in text:
                hit_match = True
            if expected_text and expected_text.lower() in aria:
                hit_match = True
            if expected_role and expected_role.lower() == role:
                hit_match = True
            pe = (info.get("pointer_events") or "").lower()
            if pe == "none":
                hit_match = False

        record = {
            "label": label,
            "x": int(x),
            "y": int(y),
            "expected_text": expected_text,
            "expected_role": expected_role,
            "actual": info,
            "hit": bool(hit_match),
            "ts_ms": _now_ms(),
        }
        if len(self._hit_tests) < MAX_HIT_TESTS:
            self._hit_tests.append(record)
        return record
