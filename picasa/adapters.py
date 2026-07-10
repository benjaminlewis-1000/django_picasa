# adapters.py
from allauth.account.adapter import DefaultAccountAdapter

class SubdomainRedirectAdapter(DefaultAccountAdapter):
    def get_login_redirect_url(self, request):
        # Look for the dynamic ?next= parameter provided by React
        next_param = request.GET.get('next') or request.POST.get('next')

        if next_param and 'facewire.exploretheworld.tech' in next_param:
            return next_param
        return super().get_login_redirect_url(request)
