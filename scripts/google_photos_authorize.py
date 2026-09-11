#! /usr/bin/env python
"""One-time local bootstrap for the "Google Photos" Tools-tab feature
(api/photos_watch_views.py) -- mints the long-lived refresh token the
backend needs to create Picker API sessions on your behalf.

Run this ONCE, on your own machine (not inside the picasa_api container --
it needs a real browser and a local port to receive Google's redirect,
neither of which the server has):

    pip install google-auth-oauthlib   # not a project dependency; only
                                        # needed to run this script locally
    python scripts/google_photos_authorize.py --client-id ... --client-secret ...

Before running this, in Google Cloud Console:
  1. Create (or reuse) a project, enable the "Google Photos Picker API".
  2. Create an OAuth 2.0 Client ID of type "Desktop app" -- this avoids
     having to register a fixed redirect URI, since run_local_server()
     picks an ephemeral local port and Google's Desktop-app client type
     permits any localhost port.
  3. The OAuth consent screen will start in "Testing" mode -- refresh
     tokens minted there expire after 7 days no matter what. Move it to
     "Production" (Console will tell you then whether
     photospicker.mediaitems.readonly needs Google's verification review)
     to get an indefinite refresh token -- the whole point of this script
     is to avoid a recurring login, so don't skip this step.

This script opens a browser, you sign in and approve access, and it prints
a refresh token to paste into the backend's .env as
GOOGLE_PHOTOS_REFRESH_TOKEN (alongside GOOGLE_PHOTOS_CLIENT_ID/
GOOGLE_PHOTOS_CLIENT_SECRET, the same values passed as --client-id/
--client-secret here).
"""

import argparse

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/photospicker.mediaitems.readonly']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client-id', required=True)
    parser.add_argument('--client-secret', required=True)
    args = parser.parse_args()

    client_config = {
        'installed': {
            'client_id': args.client_id,
            'client_secret': args.client_secret,
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://oauth2.googleapis.com/token',
            'redirect_uris': ['http://localhost'],
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    # access_type='offline' + prompt='consent' -- without both, Google
    # sometimes omits the refresh_token on a repeat authorization from an
    # account that's already granted this same client access before.
    credentials = flow.run_local_server(port=0, access_type='offline', prompt='consent')

    if not credentials.refresh_token:
        print('No refresh_token returned -- revoke this app\'s access at '
              'https://myaccount.google.com/permissions and re-run this script.')
        return

    print('\nSuccess. Add these to the backend\'s .env:\n')
    print(f'GOOGLE_PHOTOS_CLIENT_ID={args.client_id}')
    print(f'GOOGLE_PHOTOS_CLIENT_SECRET={args.client_secret}')
    print(f'GOOGLE_PHOTOS_REFRESH_TOKEN={credentials.refresh_token}')


if __name__ == '__main__':
    main()
