# Hosting the reports — one-time setup

## Why this exists

The reports used to live only on the Mac mini, served over Tailscale. That
failed: it worked from the mini itself and hung from a phone on cellular,
because the phone's request never reliably crossed the tunnel, and because
"the link works" secretly meant "the mini is awake." A link Sean bookmarks
has to work regardless of whether any particular Mac is on.

**What we're using instead: Cloudflare Pages + Cloudflare Access.**
If you already have a Cloudflare account for something else, Pages and Access
are a *different, free* product under that same account — no new vendor, no
new bill.

- **Pages** is a static file host: you push a folder of HTML, Cloudflare
  serves it from its global edge network. Nothing about it runs on, or
  depends on, the Mac mini.
- **Access** sits in front of a Cloudflare-hosted URL and blocks every
  request until the visitor proves who they are — here, by typing a
  one-time 6-digit code Cloudflare emails to a specific address. No
  password to remember, no VPN app to install, works in mobile Safari.
  Free for up to 50 users; Sean needs one.

We're deliberately using Pages' own `*.pages.dev` address rather than a
subdomain of `explainlikeimnotkorean.com`. That means **zero DNS changes**
on the domain personal-blog depends on — nothing to escalate, nothing that
can affect the blog if it's ever misconfigured.

## What Sean needs to do, once, in the Cloudflare dashboard

This part cannot be scripted — it's account-level setup gated by Sean's own
login, and letting the agent hold an all-access Cloudflare credential to do
it for him would be the exact "credential scope" problem the CISO charter
flags. Fifteen minutes, done once.

### 1. Create the Pages project

1. Go to `dash.cloudflare.com` → **Workers & Pages** → **Create** → **Pages** → **Upload assets** (not "Connect to Git" — the repo is public and we don't want reports auto-building from every commit).
2. Project name: `nfl-props`. Upload any single placeholder HTML file to finish creation (the real content ships via `scripts/publish_report.py` afterward).
3. Cloudflare hands you a URL: `https://nfl-props.pages.dev` (or `nfl-props-<hash>.pages.dev` if the short name is taken — note the exact one it gives you).

### 2. Lock it behind Access

1. Zero Trust dashboard (`one.dash.cloudflare.com`) → **Access** → **Applications** → **Add an application** → **Self-hosted**.
2. Application domain: the `pages.dev` URL from step 1.
3. Session duration: your choice — 24h is reasonable for something checked a few times a week.
4. Add a policy: Action = **Allow**, Include = **Emails** = your own address (add any other address you want to let in, e.g. a phone's iCloud-relayed one if that differs).
5. Save. From now on, hitting the `pages.dev` URL prompts for an email + a one-time code before showing anything — including the index.

### 3. Create a scoped API token for the publish script

This is the credential that goes in `.env` and gets used by the unattended
launchd jobs. Scope it tightly — it must be able to push to Pages and
nothing else, because `.env` lives on a machine that also runs other
scripts, and because if it ever leaked, "publish rights on one Pages
project" is a very different incident than "edit rights on the whole
account."

1. `dash.cloudflare.com` → profile icon → **My Profile** → **API Tokens** → **Create Token** → **Custom token**.
2. Permissions: **Account** → **Cloudflare Pages** → **Edit**. Nothing else.
3. Account Resources: limit to the one account (not "All accounts").
4. Create, copy the token immediately (Cloudflare shows it once).
5. Also grab the **Account ID** — it's on the right-hand sidebar of almost any dashboard page, a 32-character hex string.

### 4. Fill in `.env`

```
CLOUDFLARE_API_TOKEN=<the token from step 3>
CLOUDFLARE_ACCOUNT_ID=<the account id from step 3>
CF_PAGES_PROJECT=nfl-props
```

`.env` is already gitignored (verified — see `.gitignore`). Never paste
these into a commit, a chat message that gets pasted into a public issue,
or anywhere outside this file.

## The publish interface (for whatever wires the pipeline)

One command, callable from anywhere, no prompts, safe for launchd:

```bash
scripts/publish_report.py                              # push every report currently in reports/
scripts/publish_report.py report_2026_w2_DET-BUF.html   # push just the one just generated
```

Both forms regenerate `index.html` from *everything* currently in
`reports/`, so the index never drops an older week just because a given
run only mentions the newest one. It reads Cloudflare credentials from
`.env`, never prompts, and on any failure (bad token, network blip,
`wrangler` erroring) prints to stderr and exits non-zero. Run from
launchd, stderr already lands in `logs/launchd.log` per the existing
job configuration — a broken publish will show up there, not vanish.

## What happened to the old Tailscale server

`scripts/serve_reports.py` and the `com.nflprops.web` launchd job (Tailscale
Serve on the Mac mini, ports 8444/8080 → 127.0.0.1:8792) are superseded —
that's the setup that failed the phone test. I unloaded the launchd job
(`launchctl unload ~/Library/LaunchAgents/com.nflprops.web.plist`) so it
isn't sitting there half-alive; the script and its LaunchAgent plist are
left in the tree for reference rather than deleted, since something else
may still point at them. Its index-rendering logic is what
`publish_report.py` reuses above. Safe to delete both once Cloudflare Pages
has been live for a while and nothing regresses.

## The bookmark

```
https://nfl-props.pages.dev
```
(or whatever exact hostname step 1 assigned, if the short name was taken —
put the real one here once known).

Newest report first, every time, from any device, whether the Mac mini is
on or not. Cloudflare Access intercepts the request before Pages ever sees
it, so an unauthenticated visitor sees Cloudflare's login prompt, not the
report.
