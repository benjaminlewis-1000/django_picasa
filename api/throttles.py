# throttles.py
#
# Rate limits for the public, slideshow-key-gated endpoints (image_list,
# keyed_image/slideshow, keyed_video). These are the only endpoints
# reachable with nothing but the low-trust slideshow key baked into
# net_slideshow's public JS bundle - so unlike the rest of the API
# (which requires a real Authelia session), there's no real identity
# behind a request here, just an IP. Keyed by IP (SimpleRateThrottle's
# default ident) rather than the slideshow key itself, since the key is
# a single shared secret everyone with the bundle has - throttling by
# key alone would mean one scraper's activity could rate-limit the
# actual kiosk too.
#
# Real session-authenticated requests (e.g. the face-assigning
# frontend) never hit these - they bypass the slideshow-key machinery
# entirely (see KeyedImageView/KeyedVideoView's `is_authenticated`
# checks), so these limits only ever constrain the public, unauthenticated
# path.
from rest_framework.throttling import SimpleRateThrottle


class SlideshowListThrottle(SimpleRateThrottle):
    scope = 'slideshow_list'

    def get_cache_key(self, request, view):
        return self.cache_format % {
            'scope': self.scope,
            'ident': self.get_ident(request),
        }


class SlideshowMediaThrottle(SimpleRateThrottle):
    scope = 'slideshow_media'

    def get_cache_key(self, request, view):
        return self.cache_format % {
            'scope': self.scope,
            'ident': self.get_ident(request),
        }
