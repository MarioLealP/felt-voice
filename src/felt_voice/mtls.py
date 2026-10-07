"""mTLS for felt-voice's API (WP-62 M2b B5.1, decision D4).

felt-core is the only caller: she presents her 2-hour Felt service cert, and
felt-voice presents its own (``felt-voice.felt.internal``). Both come from the
laptop's renewer (clockmost infra, projects/openbao/offhost) as one
``bundle.pem`` (key + cert + intermediate) per purpose, next to ``felt-ca.pem``
(the pki_felt intermediate: the ONLY trust anchor - never the clockmost root).

- The TLS handshake requires a client cert that chains to ``felt-ca.pem``.
- On top of that, every connection's peer must carry exactly one SPIFFE URI and
  it must be in ``allowed`` (felt-core's), checked before a single request byte
  is read - any other Felt workload's cert is refused.
- The server cert is re-read when the bundle changes (renewed hourly); new
  handshakes use it, open connections keep theirs.

Off unless ``FELT_VOICE_TLS_BUNDLE`` is set (plain HTTP as before).
"""

import logging
import os
import ssl
import threading

import uvicorn
from uvicorn.protocols.http.auto import AutoHTTPProtocol

log = logging.getLogger(__name__)


def peer_uris(transport) -> list[str]:
    cert = transport.get_extra_info("peercert") or {}
    return [v for k, v in cert.get("subjectAltName", ()) if k == "URI"]


def make_protocol(allowed: frozenset[str]):
    """The HTTP protocol, refusing any peer whose identity isn't allowed."""

    class MTLSProtocol(AutoHTTPProtocol):
        def connection_made(self, transport):  # after the TLS handshake
            uris = peer_uris(transport)
            if len(uris) != 1 or uris[0] not in allowed:
                log.warning(
                    "refused TLS client %s: identity %s is not allowed",
                    transport.get_extra_info("peername"),
                    uris or "none",
                )
                self._refused = True
                transport.abort()
                return
            super().connection_made(transport)

        def connection_lost(self, exc):
            if getattr(self, "_refused", False):  # nothing was set up to tear down
                return
            super().connection_lost(exc)

    return MTLSProtocol


def harden(ctx: ssl.SSLContext) -> None:
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN  # felt-ca.pem is an intermediate


def watch_bundle(
    ctx: ssl.SSLContext, bundle: str, interval: float, stop: threading.Event
) -> None:
    """Reload the server cert into the live context whenever the renewer
    replaces the bundle."""
    seen = os.stat(bundle).st_mtime_ns
    while not stop.wait(interval):
        try:
            now = os.stat(bundle).st_mtime_ns
            if now != seen:
                ctx.load_cert_chain(bundle)
                seen = now
                log.info("reloaded the server certificate from %s", bundle)
        except (OSError, ssl.SSLError) as e:
            log.error(
                "server certificate reload failed (keeping the current one): %s", e
            )


def serve(
    app, host: str, port: int, bundle: str, ca: str, allowed, reload_s: float = 60
) -> None:
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        ssl_certfile=bundle,
        ssl_keyfile=bundle,
        ssl_ca_certs=ca,
        ssl_cert_reqs=ssl.CERT_REQUIRED,
        http=make_protocol(frozenset(allowed)),
    )
    config.load()
    harden(config.ssl)
    stop = threading.Event()
    threading.Thread(
        target=watch_bundle, args=(config.ssl, bundle, reload_s, stop), daemon=True
    ).start()
    log.info(
        "mTLS on %s:%s, allowed clients: %s", host, port, ", ".join(sorted(allowed))
    )
    try:
        uvicorn.Server(config).run()
    finally:
        stop.set()
