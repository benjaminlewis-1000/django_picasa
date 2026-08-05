# middleware.py

import re

class LocalNetworkCorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        origin = request.headers.get('Origin')

        print("ORI", origin)
        # If origin exists, check if it matches local/LAN patterns or your domain
        if origin:
            if re.match(r"^https?://(192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|localhost|127\.0\.0\.1)(:\d+)?$", origin) or "slider.exploretheworld.tech" in origin:
                response["Access-Control-Allow-Origin"] = origin
                response["Access-Control-Allow-Credentials"] = "true"
                response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
                response["Access-Control-Allow-Headers"] = "*"
        else:
            # When loaded via local IP or plain <img> tag where Origin header is omitted/None
            # Fallback to wildcard or mirror the Host header for local network ranges
            host = request.headers.get('Host', '')
            print(host)
            if any(net in host for net in ['192.168.', '10.', 'localhost', '127.0.0.1']):
                response["Access-Control-Allow-Origin"] = "*"
                response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
                response["Access-Control-Allow-Headers"] = "*"

        return response