// Service worker for the Match Predictor PWA.
// Strategy: cache the app shell (HTML/CSS/JS/icons) so the app opens
// instantly and works offline for browsing the UI. API calls (/api/*)
// are ALWAYS network-first — predictions depend on live data and the
// current model, so serving a stale cached prediction would be actively
// wrong, not just inconvenient. If the network fails for an API call,
// this deliberately lets the request fail rather than returning stale data.

const CACHE_NAME = "match-predictor-v1";
const APP_SHELL = [
  "/",
  "/manifest.json",
  "/icon-192.png",
  "/icon-512.png",
  "/static/football-hero.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Never cache API calls — always go to the network.
  if (url.pathname.startsWith("/api/")) {
    return; // let the browser handle it normally, no caching involved
  }

  // App shell: cache-first, falling back to network, and updating the cache in the background.
  event.respondWith(
    caches.match(event.request).then((cached) => {
      const networkFetch = fetch(event.request)
        .then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => cached); // offline and not cached -> nothing we can do for this resource

      return cached || networkFetch;
    })
  );
});
