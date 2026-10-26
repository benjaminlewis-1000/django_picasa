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
# Real session-authenticated requests never get throttled at all --
# get_cache_key() returning None is DRF's documented "skip this
# throttle entirely" signal. This was previously only a claim in this
# comment, not actually implemented (confirmed real 2026-10-26: a
# session-authenticated frontend burst-loading face thumbnails via
# KeyedImageView's access_key-gated types was getting 429'd exactly
# like an anonymous caller would, since get_cache_key() never checked
# request.user.is_authenticated at all) -- now it's real.
from rest_framework.throttling import SimpleRateThrottle


class SlideshowListThrottle(SimpleRateThrottle):
    scope = 'slideshow_list'

    def get_cache_key(self, request, view):
        if request.user.is_authenticated:
            return None
        return self.cache_format % {
            'scope': self.scope,
            'ident': self.get_ident(request),
        }


class SlideshowMediaThrottle(SimpleRateThrottle):
    """The sig-gated media path (KeyedImageView's type='slideshow',
    KeyedVideoView) -- a per-photo/per-video, time-boxed token, not a
    shared secret. This is the real slideshow kiosk's own normal,
    continuous traffic, so it gets the more generous rate."""
    scope = 'slideshow_media'

    def get_cache_key(self, request, view):
        if request.user.is_authenticated:
            return None
        return self.cache_format % {
            'scope': self.scope,
            'ident': self.get_ident(request),
        }


class AccessKeyMediaThrottle(SimpleRateThrottle):
    """KeyedImageView's other types (face_array/face_source/
    face_highlight/full_small) when accessed unauthenticated, via the
    shared, unscoped RANDOM_ACCESS_KEY rather than a per-item sig --
    lower trust than the sig-gated path above (one leaked key grants
    access to every face/image, not just one), so a tighter rate."""
    scope = 'access_key_media'

    def get_cache_key(self, request, view):
        if request.user.is_authenticated:
            return None
        return self.cache_format % {
            'scope': self.scope,
            'ident': self.get_ident(request),
        }
