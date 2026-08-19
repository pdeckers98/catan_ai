"""The action sender: colonist frames, synthesized rather than clicked.

Everything else in ``src/bridge/`` reads. This module writes, and it is the one
piece no amount of captured traffic can settle on its own -- the format is
legible, but whether the server accepts a frame the page's own client did not
originate is only answerable by sending one.

**Two halves, deliberately separable.** :class:`FrameCodec` turns a move into the
bytes colonist's client would have emitted, and is pure: it is tested
byte-for-byte against every in-game frame of a real capture, offline, with no
browser. :class:`PageSender` puts those bytes on the wire, by reaching into the
page and using colonist's *own* socket -- so the server sees the authenticated
connection it already trusts, not a second one.

**Why the page's socket and not our own.** A second connection would have to
re-authenticate, would show up as a duplicate session, and is the version most
likely to look like a bot. CDP has no command for writing a websocket frame, so
the socket has to be reached from inside the page: an init script wraps
``window.WebSocket`` before colonist opens it, keeps the instances, and exposes
a send hook. This costs nothing at read time and keeps the single-session design
the rest of the bridge is built around.

**The header was unknown when the doc was written and is not any more.** Client
frames carry ``<kind> <channel> <len> <room>`` before the msgpack body. Over a
full live game the in-game header was byte-identical on all 188 frames
(``03 01 06 <game id>``), while the lobby used ``02 07 05 "lobby"``. That is
enough to hardcode -- and it is not hardcoded anyway. :class:`FrameCodec` learns
the header by *watching* the client send one, so a frame we synthesize is routed
exactly the way the client routed its own, whatever the bytes turn out to mean.

⚠️ Sending moves is what makes this phase a ToS problem rather than a curiosity.
Throwaway account, supervised, never ranked. Nothing here runs unless asked for.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import msgpack
except ImportError:  # pragma: no cover - exercised by the import guard alone
    msgpack = None


# --------------------------------------------------------------------------
# What the client sends. Correlated against the log entries each frame produced,
# over three captured games; see docs/PHASE3_WEB.md for the derivation.
# --------------------------------------------------------------------------

SEND_ROLL = 2                 # payload: true
SEND_END_TURN = 6             # payload: true
SEND_RESOLVE_CARDS = 7        # payload: [card, ...]   (a discard)
SEND_RESOLVE_CARD = 8         # payload: [card]        (monopoly / year of plenty)
SEND_INITIAL_ROAD = 11        # payload: edge id
SEND_ROAD = 12                # payload: edge id
SEND_INITIAL_SETTLEMENT = 15  # payload: corner id
SEND_SETTLEMENT = 16          # payload: corner id
SEND_CITY = 19                # payload: corner id
SEND_MOVE_ROBBER = 3          # payload: tile index
SEND_PLAY_DEV_CARD = 48       # payload: card enum
SEND_TRADE = 49               # payload: {creator, isBankTrade, ...}

#: Frames that are safe to send as a probe: they either succeed at a moment we
#: chose or are refused, and neither outcome costs a position.
PROBE_ACTIONS = {"roll": (SEND_ROLL, True), "end": (SEND_END_TURN, True)}

#: The room kinds seen in a header's first byte.
ROOM_LOBBY = 2
ROOM_GAME = 3


class SendError(RuntimeError):
    """A frame could not be built, or the page would not put it on the wire."""


@dataclass(frozen=True)
class RoutingHeader:
    """The bytes colonist puts in front of a client frame's msgpack body.

    ``kind`` distinguishes the lobby from a game room, ``channel`` is a small
    integer that is constant for the life of a room, and ``room`` is the room
    name -- "lobby", or the game id in play.
    """

    kind: int
    channel: int
    room: str

    def encode(self) -> bytes:
        name = self.room.encode("ascii")
        return bytes([self.kind, self.channel, len(name)]) + name

    @classmethod
    def parse(cls, raw: bytes) -> "RoutingHeader":
        if len(raw) < 4:
            raise SendError(f"routing header too short: {raw.hex()}")
        length = raw[2]
        name = raw[3:]
        if length != len(name):
            raise SendError(f"routing header length {length} != {len(name)}: {raw.hex()}")
        return cls(kind=raw[0], channel=raw[1], room=name.decode("ascii"))


def encode_frame(header: RoutingHeader, action: int, payload: Any,
                 sequence: Optional[int] = None) -> bytes:
    """Build one client frame: routing header, then the msgpack body.

    Key order matters only in that it must round-trip; colonist's client emits
    ``action``, ``payload``, ``sequence`` in that order and so do we, which is
    what lets :func:`frames_round_trip` compare against captured bytes rather
    than merely against a re-decode.
    """
    if msgpack is None:
        raise SendError("msgpack is required to send frames: pip install msgpack")
    body: Dict[str, Any] = {"action": action, "payload": payload}
    if sequence is not None:
        body["sequence"] = sequence
    return header.encode() + msgpack.packb(body, use_bin_type=True)


@dataclass
class FrameCodec:
    """Learns a room's routing from the client, then speaks in its voice.

    Two things have to be right for a synthesized frame to be indistinguishable
    from the client's own: the routing header, and ``sequence``. Both are read
    off the frames the client is already sending -- which the live session
    records anyway -- rather than assumed.

    ``sequence`` is the sharp one. It increments per action, and the client has
    no idea we consumed a number, so after we send, the client's next frame will
    reuse it. Whether the server minds is exactly the kind of thing the first
    live send is for; :attr:`sent` records what we spent so it can be read back
    against what happened.
    """

    header: Optional[RoutingHeader] = None
    last_sequence: Optional[int] = None
    sent: List[dict] = field(default_factory=list)

    def observe(self, entry: dict) -> None:
        """Feed one recorded ``sent`` frame, as the capture writes it."""
        if entry.get("dir") != "sent" or "header" not in entry:
            return
        try:
            header = RoutingHeader.parse(bytes.fromhex(entry["header"]))
        except (SendError, ValueError):
            return
        if header.kind != ROOM_GAME:
            return  # the lobby routes differently and we never play there
        self.header = header
        payload = entry.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("sequence"), int):
            sequence = payload["sequence"]
            if self.last_sequence is None or sequence > self.last_sequence:
                self.last_sequence = sequence

    @property
    def ready(self) -> bool:
        """True once the client has shown us how this game's frames are routed."""
        return self.header is not None and self.last_sequence is not None

    def build(self, action: int, payload: Any) -> bytes:
        if not self.ready:
            raise SendError(
                "no game frame observed yet -- the routing header and sequence are "
                "read off the client, so a move can only be sent once it has sent one"
            )
        assert self.header is not None and self.last_sequence is not None
        self.last_sequence += 1
        frame = encode_frame(self.header, action, payload, self.last_sequence)
        self.sent.append({"action": action, "payload": payload,
                          "sequence": self.last_sequence, "raw": frame.hex()})
        return frame


def frames_round_trip(records: List[dict]) -> List[dict]:
    """Re-encode captured client frames and return the ones that differ.

    The point of comparing against raw bytes rather than a re-decode: a decode
    that agrees proves our *reading* is consistent, while identical bytes prove
    the server cannot tell our frame from the client's. Anything returned here
    is a genuine difference and worth reading before sending anything live.
    """
    mismatches = []
    for entry in records:
        if entry.get("dir") != "sent" or "header" not in entry or "raw" not in entry:
            continue
        payload = entry.get("payload")
        if not isinstance(payload, dict) or "action" not in payload:
            continue
        header = RoutingHeader.parse(bytes.fromhex(entry["header"]))
        rebuilt = encode_frame(header, payload["action"], payload.get("payload"),
                               payload.get("sequence"))
        original = base64.b64decode(entry["raw"])
        if rebuilt != original:
            mismatches.append({"payload": payload, "original": original.hex(),
                               "rebuilt": rebuilt.hex()})
    return mismatches


# --------------------------------------------------------------------------
# Putting the bytes on the wire.
# --------------------------------------------------------------------------

#: Installed before any of colonist's own scripts run, so the wrapper is in
#: place by the time the socket is opened. A ``Proxy`` over the constructor is
#: used rather than a wrapper function so instances keep the real prototype and
#: the page cannot tell it has been patched by looking. ``send`` is wrapped only
#: to timestamp it: a page holds several sockets and the one carrying the game
#: is the one it most recently spoke on.
INIT_SCRIPT = """
(() => {
  if (window.__colonistSend) return;
  const Original = window.WebSocket;
  const sockets = [];
  window.__colonistSockets = sockets;
  window.WebSocket = new Proxy(Original, {
    construct(target, args) {
      const ws = new target(...args);
      if (String(args[0] || '').includes('colonist')) {
        ws.__lastSend = 0;
        const send = ws.send.bind(ws);
        ws.send = function (data) { ws.__lastSend = Date.now(); return send(data); };
        sockets.push(ws);
      }
      return ws;
    }
  });
  window.__colonistSend = (b64) => {
    const open = sockets.filter((s) => s.readyState === 1);
    if (!open.length) return {ok: false, why: 'no open colonist socket'};
    open.sort((a, b) => (a.__lastSend || 0) - (b.__lastSend || 0));
    const ws = open[open.length - 1];
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    try {
      ws.send(bytes);
    } catch (err) {
      return {ok: false, why: String(err)};
    }
    return {ok: true, sockets: open.length, bytes: bytes.length};
  };
})();
"""


class PageSender:
    """Writes frames on colonist's own socket, from inside the page.

    Constructed around a Playwright page whose context had :data:`INIT_SCRIPT`
    added *before* navigation -- see :func:`install`. If the script was added
    late the hook is simply absent and :meth:`send` says so, rather than failing
    somewhere subtler.
    """

    def __init__(self, page, codec: Optional[FrameCodec] = None):
        self.page = page
        self.codec = codec or FrameCodec()

    def observe(self, entry: dict) -> None:
        self.codec.observe(entry)

    @property
    def ready(self) -> bool:
        return self.codec.ready

    def send_frame(self, frame: bytes) -> dict:
        encoded = base64.b64encode(frame).decode()
        try:
            result = self.page.evaluate(
                "(b64) => window.__colonistSend ? window.__colonistSend(b64)"
                " : {ok: false, why: 'hook missing -- init script ran too late'}",
                encoded,
            )
        except Exception as exc:  # the page navigated or closed under us
            raise SendError(f"page.evaluate failed: {exc}") from exc
        if not result or not result.get("ok"):
            raise SendError(str((result or {}).get("why", "unknown send failure")))
        return result

    def send(self, action: int, payload: Any) -> dict:
        """Build and send one move. Returns what the page reported."""
        frame = self.codec.build(action, payload)
        result = self.send_frame(frame)
        result["sequence"] = self.codec.last_sequence
        return result

    def send_probe(self, name: str) -> dict:
        """Send one of :data:`PROBE_ACTIONS` -- the frames that cost nothing."""
        if name not in PROBE_ACTIONS:
            raise SendError(f"unknown probe {name!r}; try one of {sorted(PROBE_ACTIONS)}")
        action, payload = PROBE_ACTIONS[name]
        return self.send(action, payload)

    def send_raw(self, action: int, payload_json: str) -> dict:
        """Send an arbitrary action, payload given as JSON. For hand testing."""
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            raise SendError(f"payload is not JSON: {exc}") from exc
        return self.send(action, payload)


def install(context) -> None:
    """Add the socket hook to a browser context, before anything navigates."""
    context.add_init_script(INIT_SCRIPT)
