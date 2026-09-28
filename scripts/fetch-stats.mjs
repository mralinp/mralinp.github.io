// Fetch the numbers for the "At a glance" panels and write _data/stats.json, which the templates
// read at build time. Runs in the Pages workflow before `jekyll build` (and daily on a schedule),
// so visitors never wait on, or get tracked by, these services.
//
// Any source that fails keeps its last committed value: a GitHub or ORCID outage leaves the site
// a day stale, never showing zeros. No dependencies; Node 18+ for fetch.
//
//   GITHUB_TOKEN=... node scripts/fetch-stats.mjs
import { readFileSync, writeFileSync } from "node:fs";

const FILE = new URL("../_data/stats.json", import.meta.url);
const USER = "mralinp";
const ORCID = "0009-0004-6491-8950";
const S2 = "2385183618";
const token = process.env.GITHUB_TOKEN;

let old = {};
try { old = JSON.parse(readFileSync(FILE, "utf8")); } catch {}
const out = { updated: old.updated, github: { ...old.github }, academic: { ...old.academic } };
let changed = false;

async function json(url, opts = {}) {
  const r = await fetch(url, { ...opts, headers: { "User-Agent": "alinaderiparizi.com stats", ...(opts.headers || {}) } });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
async function step(name, fn) {
  try { await fn(); changed = true; console.log(`ok   ${name}`); }
  catch (e) { console.warn(`keep ${name} (${e.message})`); }
}
const gh = token ? { Authorization: `Bearer ${token}` } : {};

await step("github profile and repos", async () => {
  const user = await json(`https://api.github.com/users/${USER}`, { headers: gh });
  const repos = [];
  for (let page = 1; ; page++) {
    const batch = await json(`https://api.github.com/users/${USER}/repos?per_page=100&page=${page}`, { headers: gh });
    repos.push(...batch);
    if (batch.length < 100) break;
  }
  const own = repos.filter((r) => !r.fork);
  const langs = {};
  for (const r of own) if (r.language) langs[r.language] = (langs[r.language] || 0) + 1;
  Object.assign(out.github, {
    repos: user.public_repos,
    followers: user.followers,
    stars: own.reduce((n, r) => n + r.stargazers_count, 0),
    languages: Object.entries(langs).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([l]) => l),
  });
});

await step("github contributions", async () => {
  if (!token) throw new Error("no GITHUB_TOKEN");
  const LEVEL = { NONE: 0, FIRST_QUARTILE: 1, SECOND_QUARTILE: 2, THIRD_QUARTILE: 3, FOURTH_QUARTILE: 4 };
  const q = `{ user(login: "${USER}") { contributionsCollection { contributionCalendar {
      totalContributions weeks { contributionDays { contributionLevel } } } } } }`;
  const r = await json("https://api.github.com/graphql", { method: "POST", headers: gh, body: JSON.stringify({ query: q }) });
  const cal = r.data.user.contributionsCollection.contributionCalendar;
  out.github.contributions = cal.totalContributions;
  out.github.calendar = cal.weeks.flatMap((w) => w.contributionDays.map((d) => LEVEL[d.contributionLevel] ?? 0));
});

await step("semantic scholar", async () => {
  const a = await json(`https://api.semanticscholar.org/graph/v1/author/${S2}?fields=citationCount,hIndex`);
  // Semantic Scholar indexes less than Google Scholar (which has no API), and these counts only
  // grow, so never go below the stored value; seed _data/stats.json with Scholar's numbers.
  Object.assign(out.academic, {
    citations: Math.max(a.citationCount, out.academic.citations || 0),
    h_index: Math.max(a.hIndex, out.academic.h_index || 0),
  });
});

await step("orcid", async () => {
  const h = { headers: { Accept: "application/json" } };
  const works = await json(`https://pub.orcid.org/v3.0/${ORCID}/works`, h);
  const reviews = await json(`https://pub.orcid.org/v3.0/${ORCID}/peer-reviews`, h);
  // a review group per journal; count the individual reviews inside them
  const nReviews = (reviews.group || []).reduce((n, g) => n + (g["peer-review-group"] || []).reduce((m, pg) => m + (pg["peer-review-summary"] || []).length, 0), 0);
  Object.assign(out.academic, { publications: (works.group || []).length, reviews: nReviews });
});

if (changed) out.updated = new Date().toISOString().slice(0, 10);
writeFileSync(FILE, JSON.stringify(out, null, 2) + "\n");
console.log(`wrote ${FILE.pathname}`);
