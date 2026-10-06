# Real incident, 2026-10-05: found while investigating an unrelated memory-
# growth question in gunicorn's own workers -- a vulnerability-scanning bot
# (routed through the Cloudflare Tunnel, so its real IP isn't visible to us,
# just the tunnel's own internal address) constantly probes this server with
# hundreds of classic WordPress/PHP exploit-probe paths per minute
# (/wp-signup.php, /xmlrpc.php, /adminfuns.php, etc.) -- ordinary internet
# background noise any exposed web server gets, not something specific to
# this app. This Django app never serves a literal .php file, ever, so
# rejecting any such path immediately -- before CORS handling, session
# lookup, auth, or URL resolution -- is unconditionally safe and makes
# handling this traffic nearly free instead of running it through the full
# request pipeline just to 404.
#
# Real fix for the underlying noise (this traffic reaching the server at
# all) is Cloudflare's own Bot Fight Mode, which stops it at the edge --
# outside what this app can control. This middleware only makes the part
# that *does* reach us cheap, not a replacement for that.

SCANNED_EXTENSIONS = ('.php', '.php3', '.php4', '.php5', '.php7', '.phtml', '.asp', '.aspx', '.cgi')


class BlockScannerPathsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.lower().endswith(SCANNED_EXTENSIONS):
            from django.http import HttpResponseForbidden
            return HttpResponseForbidden('Not found.')
        return self.get_response(request)
