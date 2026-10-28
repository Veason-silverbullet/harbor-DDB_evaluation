Build a complete, working web application for this request:

{{QUERY}}

Work in `/app`. Node.js, npm, Python, and Chromium are installed. You can install
additional dependencies. Implement the requested behavior, including persistent
storage and a backend when needed. Use real online data where the request calls
for it.

Create `/app/start.sh` that starts the application on `0.0.0.0:5173` and stays
running in the foreground. It must work when invoked with `bash /app/start.sh`
from `/app`, including starting any required backend services. Install dependencies
before finishing. Verify the app in a browser.

After verification, stop the app servers and finish any background commands before
ending your response. The evaluator will start the app using `/app/start.sh`.

For collaborative features, independent users will open the app in separate
browser sessions. Provide the requested login, sharing, and identity flows in
the app. If participants need distinct entry URLs, write `/app/session_urls.json`
mapping participant names (such as `viewer_a`, `viewer_b`) to local URLs or paths.
These are entry hints, not pre-authenticated identities. The browser evaluator
will discover invitations through the app, perform signup/login and join flows
as needed, and keep each participant's separate session open for follow-up actions.
