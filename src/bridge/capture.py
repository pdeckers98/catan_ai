"""Record colonist.io's WebSocket traffic from a browser you drive by hand.

This is the read half of the transport, run standalone. It opens a real browser
window, you log in and play (or spectate) a game yourself, and every WebSocket
frame in both directions is written to a JSONL file. Nothing is automated and no
click is ever sent -- the point is to *observe* the protocol, which is
undocumented, so that the translator in Phase 3 can be written against recorded
evidence instead of guesses.

Two reasons this is Playwright rather than Selenium, beyond the transport
decision already recorded in ``docs/PHASE3_WEB.md``: Playwright exposes a raw
CDP session, so frames are read from the network layer rather than by injecting
script into the page, and the same page handle later does the clicking. Selenium
has no first-class equivalent of either.

The browser profile is persistent (``data/bridge/profile``), so a login survives
between runs -- log in once, capture as many games as you like.

Usage::

    python -m src.bridge.capture --label game1     # capture until you close it
    python -m src.bridge.capture --summarize data/bridge/game1-<stamp>.jsonl

The summarize mode is the one that matters for analysis: a full game is
megabytes of frames, and what a reader needs first is the *shape* -- which
message types exist, how often, and one example of each.

⚠️ Use a throwaway account. Automating colonist.io likely violates its Terms of
Service; this module only reads, but the account you log in with is the one that
will later be driven.
"""

import argparse
import base64
import binascii
import json
import re
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

try:
    import msgpack
except ImportError:  # pragma: no cover - optional, only some frames need it
    msgpack = None

DEFAULT_URL = "https://colonist.io"
CAPTURE_DIR = Path("data/bridge")
PROFILE_DIR = CAPTURE_DIR / "profile"

# socket.io/engine.io text frames are a small integer prefix followed by JSON,
# e.g. ``42["event",{...}]``. Worth unwrapping: the prefix is transport
# bookkeeping and would otherwise make every frame decode as opaque text.
_SOCKET_IO_PREFIX = re.compile(r"^(\d+)(?=[\[{])")

# --------------------------------------------------------------------------
# Frame decoding
# --------------------------------------------------------------------------


def _framed_msgpack(raw: bytes) -> dict:
    """Decode a colonist frame that carries a routing header before its body.

    Observed on every frame the *client* sends: ``0x02 <id> <len> <name>``
    followed by msgpack, where ``name`` is the room ("lobby", and presumably the
    game id in play). Frames the server sends have been bare msgpack so far.

    Rather than hardcode a three-byte header from a handful of samples, the body
    is found by scanning for the first offset that msgpack accepts *exactly* --
    msgpack is self-delimiting, so a decode with no trailing bytes is strong
    evidence of the right offset. The skipped bytes are kept verbatim, because
    the header's meaning is still unknown: the second byte varies while the room
    name does not, and only more captures will say what it counts.
    """
    if msgpack is None:
        return {}
    for offset in range(1, 24):
        if offset >= len(raw):
            break
        try:
            body = msgpack.unpackb(raw[offset:], raw=False, strict_map_key=False)
        except Exception:
            continue
        header = raw[:offset]
        record = {"encoding": "msgpack+header", "payload": body,
                  "header": header.hex(), "raw": base64.b64encode(raw).decode()}
        # ``<len> <name>`` at the tail of the header, if that is what it is.
        length = header[2] if len(header) > 2 else -1
        name = header[3:]
        if length == len(name) and name.isascii() and name.decode().isprintable():
            record["channel"] = name.decode()
        return record
    return {}


def decode_payload(payload: str, opcode: int) -> dict:
    """Best-effort decode of one CDP frame payload.

    Returns a record with ``encoding`` naming what worked, so a later reader can
    tell a decoded structure from a blob that resisted every attempt. CDP hands
    binary frames (opcode 2) over base64-encoded and text frames (opcode 1) as
    they are.

    Undecodable frames are kept, not dropped. A frame nobody could parse is
    evidence too -- and colonist.io is at liberty to change encodings.
    """
    if opcode == 2:
        try:
            raw = base64.b64decode(payload)
        except (binascii.Error, ValueError):
            return {"encoding": "opaque", "payload": payload}
        if msgpack is not None:
            try:
                return {"encoding": "msgpack",
                        "payload": msgpack.unpackb(raw, raw=False, strict_map_key=False)}
            except Exception:
                pass
        framed = _framed_msgpack(raw)
        if framed:
            return framed
        try:
            return {"encoding": "json", "payload": json.loads(raw.decode("utf-8"))}
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        return {"encoding": "base64", "payload": payload}

    prefix = None
    body = payload
    match = _SOCKET_IO_PREFIX.match(payload)
    if match:
        prefix, body = match.group(1), payload[match.end():]
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return {"encoding": "text", "payload": payload}
    record = {"encoding": "json", "payload": decoded}
    if prefix is not None:
        record["socketio_prefix"] = prefix
    return record


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------


def capture(url: str, label: str, channel: str, out_dir: Path) -> Path:
    """Open a browser and log every WebSocket frame until it is closed."""
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{label}-{stamp}.jsonl"

    started = time.time()
    counts = Counter()

    with out_path.open("w", encoding="utf-8") as handle:

        def write(record: dict) -> None:
            record["t"] = round(time.time() - started, 3)
            handle.write(json.dumps(record, default=str) + "\n")
            # Flushed per frame so the file can be read while the session is
            # still open -- a capture is often inspected mid-game.
            handle.flush()

        def attach(page) -> None:
            cdp = page.context.new_cdp_session(page)
            cdp.send("Network.enable")

            def on_created(event):
                write({"kind": "socket", "id": event.get("requestId"),
                       "url": event.get("url")})
                print(f"  websocket opened: {event.get('url')}")

            def frame(direction):
                def handler(event):
                    response = event.get("response", {})
                    opcode = response.get("opcode", 1)
                    record = {"kind": "frame", "dir": direction,
                              "id": event.get("requestId"), "opcode": opcode}
                    record.update(decode_payload(response.get("payloadData", ""), opcode))
                    write(record)
                    counts[direction] += 1
                    if sum(counts.values()) % 100 == 0:
                        print(f"  {counts['recv']} received / {counts['sent']} sent",
                              end="\r", flush=True)
                return handler

            cdp.on("Network.webSocketCreated", on_created)
            cdp.on("Network.webSocketFrameReceived", frame("recv"))
            cdp.on("Network.webSocketFrameSent", frame("sent"))

        with sync_playwright() as driver:
            context = driver.chromium.launch_persistent_context(
                user_data_dir=str(PROFILE_DIR),
                channel=channel or None,
                headless=False,
                viewport=None,
                args=["--start-maximized"],
            )
            context.on("page", attach)
            page = context.pages[0] if context.pages else context.new_page()
            attach(page)
            page.goto(url)

            print(f"recording to {out_path}")
            print("log in and play a game; close the browser window when done "
                  "(Ctrl-C here also works)")
            try:
                while context.pages:
                    live = context.pages[0]
                    if live.is_closed():
                        continue
                    live.wait_for_timeout(500)
            except KeyboardInterrupt:
                print("\ninterrupted")
            except Exception as exc:  # the window was closed mid-wait
                if "closed" not in str(exc).lower():
                    raise
            finally:
                try:
                    context.close()
                except Exception:
                    pass

    print(f"\n{counts['recv']} received / {counts['sent']} sent -> {out_path}")
    return out_path


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


_TYPE_KEYS = ("type", "action", "event", "id")


def _discriminator(value, prefix: str = "", depth: int = 0) -> str:
    """Find the field that distinguishes one message of a kind from another.

    Only *scalar* values are usable as a label: a nested envelope like
    ``{"action": {"type": 2, ...}}`` names its kind one level down, and taking
    the whole sub-dict instead would give every message its own group and defeat
    the summary. So a dict-valued key is descended into rather than printed.
    """
    if not isinstance(value, dict) or depth >= 3:
        return ""
    for key in _TYPE_KEYS:
        if key not in value:
            continue
        inner = value[key]
        if isinstance(inner, dict):
            found = _discriminator(inner, f"{prefix}{key}.", depth + 1)
            if found:
                return found
            continue
        return f"{prefix}{key}={inner!r}"
    return ""


def message_type(payload) -> str:
    """A short label for one decoded message, used to group the summary.

    The protocol is unknown, so this guesses rather than knows: socket.io
    payloads are ``[event, data]``, and colonist's own envelopes have carried a
    ``type`` or ``id`` field in the prior art. Anything unrecognised is grouped
    by its top-level key set, which is still far more informative than one
    undifferentiated pile.
    """
    if isinstance(payload, list) and payload and isinstance(payload[0], str):
        head = payload[0]
        rest = payload[1] if len(payload) > 1 else None
        found = _discriminator(rest)
        return f"{head}/{found}" if found else head
    if isinstance(payload, dict):
        return _discriminator(payload) or (
            "{" + ",".join(sorted(map(str, payload))[:6]) + "}")
    if isinstance(payload, list):
        return f"list[{len(payload)}]"
    return type(payload).__name__


def shape(value, depth: int = 0, max_depth: int = 3):
    """Recursive key structure of a message, with the values thrown away.

    Two messages of the same type usually differ only in their values; showing
    the shape once and one concrete example is the difference between reading a
    protocol and reading a transcript.
    """
    if depth >= max_depth:
        return "..."
    if isinstance(value, dict):
        return {str(k): shape(v, depth + 1, max_depth) for k, v in list(value.items())[:24]}
    if isinstance(value, list):
        if not value:
            return []
        # Short lists are usually tuples of unrelated things -- a socket.io
        # frame is ``[event_name, data]`` -- so collapsing them to "first
        # element x N" would hide the payload entirely. Long ones are
        # homogeneous and the first element speaks for the rest.
        if len(value) <= 4:
            return [shape(item, depth + 1, max_depth) for item in value]
        return [shape(value[0], depth + 1, max_depth), f"x{len(value)}"]
    return type(value).__name__


def summarize(path: Path, limit: int, chars: int, only: str, socket: str) -> None:
    sockets = {}
    groups = OrderedDict()
    counts = Counter()
    encodings = Counter()
    skipped = Counter()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("kind") == "socket":
                sockets[record.get("id")] = record.get("url")
                continue
            if record.get("kind") != "frame":
                continue
            # A browser has other sockets open -- Discord's login gateway and a
            # dozen localhost RPC ports showed up in the first capture. Keep
            # them on disk, filter them out of the reading.
            url = sockets.get(record.get("id"), "")
            if socket and socket.lower() not in url.lower():
                skipped[url or "?"] += 1
                continue
            # Frames recorded before the decoder understood a given framing are
            # re-decoded here, so an old capture improves without being replayed.
            if record.get("encoding") == "base64":
                record = {**record, **decode_payload(record["payload"], 2)}
            counts[record["dir"]] += 1
            encodings[record.get("encoding", "?")] += 1
            payload = record.get("payload")
            channel = record.get("channel")
            label = f"{record['dir']} {'[' + channel + '] ' if channel else ''}" \
                    f"{message_type(payload)}"
            if only and only.lower() not in label.lower():
                continue
            entry = groups.setdefault(label, {"n": 0, "example": payload, "t": record.get("t")})
            entry["n"] += 1

    print(f"# {path}")
    print(f"frames: {counts['recv']} received, {counts['sent']} sent")
    print(f"encodings: {dict(encodings)}")
    for socket_id, url in sockets.items():
        if not socket or socket.lower() in url.lower():
            print(f"socket {socket_id}: {url}")
    if skipped:
        print(f"skipped {sum(skipped.values())} frames on other sockets "
              f"({len(skipped)} of them); pass --socket '' to include them")
    print(f"\n{len(groups)} distinct message types "
          f"(showing the {min(limit, len(groups))} most frequent)\n")

    ordered = sorted(groups.items(), key=lambda kv: -kv[1]["n"])[:limit]
    for label, entry in ordered:
        print(f"--- {label}  x{entry['n']}  (first at t={entry['t']}s)")
        print(f"    shape:   {json.dumps(shape(entry['example']), default=str)[:chars]}")
        print(f"    example: {json.dumps(entry['example'], default=str)[:chars]}")
        print()


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--summarize", metavar="FILE",
                        help="analyse an existing capture instead of recording")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--label", default="capture",
                        help="filename prefix for this recording")
    parser.add_argument("--channel", default="chrome",
                        help="browser channel; 'chrome' uses your installed Chrome, "
                             "empty string uses Playwright's bundled Chromium")
    parser.add_argument("--out-dir", type=Path, default=CAPTURE_DIR)
    parser.add_argument("--limit", type=int, default=40,
                        help="summarize: message types to print")
    parser.add_argument("--chars", type=int, default=600,
                        help="summarize: characters per example")
    parser.add_argument("--only", default="",
                        help="summarize: keep only message types containing this text")
    parser.add_argument("--socket", default="colonist",
                        help="summarize: only frames on sockets whose URL contains this; "
                             "pass an empty string for every socket the browser opened")
    args = parser.parse_args()

    if args.summarize:
        summarize(Path(args.summarize), args.limit, args.chars, args.only, args.socket)
        return 0

    capture(args.url, args.label, args.channel, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
