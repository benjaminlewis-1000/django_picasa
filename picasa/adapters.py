# adapters.py
import re
from urllib.parse import urlparse

from allauth.account.adapter import DefaultAccountAdapter

# Matches http:// or https:// for any exploretheworld.tech subdomain --
# same pattern picasa/settings.py's CORS_ALLOWED_ORIGIN_REGEXES uses.
_ALLOWED_REDIRECT_HOST_RE = re.compile(r'^([a-zA-Z0-9_-]+\.)*exploretheworld\.tech$')


class SubdomainRedirectAdapter(DefaultAccountAdapter):
    def get_login_redirect_url(self, request):
        # Look for the dynamic ?next= parameter provided by React
        next_param = request.GET.get('next') or request.POST.get('next')

        if next_param:
            parsed = urlparse(next_param)
            # A relative path (no scheme/host) can only ever resolve
            # against this same origin -- safe by construction.
            if not parsed.netloc:
                return next_param
            # An absolute URL must resolve to a real exploretheworld.tech
            # subdomain. This checks the actual parsed hostname, not
            # whether the string contains the domain name anywhere --
            # `next=https://evil.example/?x=facewire.exploretheworld.tech`
            # used to pass the old plain substring check and redirect a
            # freshly-authenticated user to an attacker-controlled host.
            if parsed.scheme in ('http', 'https') and _ALLOWED_REDIRECT_HOST_RE.match(parsed.hostname or ''):
                return next_param

        return super().get_login_redirect_url(request)
