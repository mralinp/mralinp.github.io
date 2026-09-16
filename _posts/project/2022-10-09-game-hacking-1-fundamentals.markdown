---
layout: post
title:  "Game Hacking, Part 1: Anti-Cheats, Engines, and How Cheats Actually Work"
author: "Ali N. Parizi"
img: "/assets/images/posts/projects/csgo-cheat/csgo-logo-d.jpg"
date:   2022-10-09 12:21:13 +0330
categories:  project security cracking game-hacking
brief: "Anti-cheat, game engines, internal vs. external cheats, injection, and memory reading and writing — the concepts behind every game cheat, before we build one."
---
This is the first of two posts. This one is background: the concepts behind how game cheats and anti-cheats actually work. The second is a report on [khiar-ware](https://github.com/mralinp/khiar-ware), a CS:GO cheat I built to learn this material hands-on.

**This project and these posts are for educational purposes only**, to understand game security from the offense side — the same way people study exploit development to understand defense. Cheating in an online game against real opponents ruins the game for them and gets accounts banned (Valve's VAC, in particular, does not forgive). Everything demonstrated in the next post was run offline, against bots, with the game's anti-cheat explicitly disabled for testing. If you build on this, do the same.

# 1. What a cheat actually needs

Strip away the specifics of any given game, and a cheat needs exactly two things: **information** the game has and isn't showing you (where the enemies are, whether they're behind a wall, how much ammo they have), and **influence** over what happens next (aim precisely, fire exactly on target, jump at the perfect frame). Nearly everything that gets called a "cheat" is one or both of those: an ESP/wallhack is pure information — read the enemy's position and draw it, or make the game itself render it through walls. An aimbot or triggerbot is information plus influence — read where an enemy is, then move the mouse or fire a shot for you. A speedhack or bunnyhop bot is closer to pure influence — nudge a value or a timed input the game wasn't expecting.

Both of those — information and influence — come from the same place: the game process's memory. A running game is, from the operating system's point of view, just a process with a big chunk of memory holding the current state of the world: player positions, health, view angles, weapon state. If you can read that memory, you have information. If you can write to it, or feed the game inputs at the right moment, you have influence.

# 2. What a game engine keeps in memory (roughly)

You don't need the source code to guess the shape of what's in there, because most game engines solve the same problem the same way. There's typically an **entity list** — an array or table of every object in the current scene, players and bots included, each with a class identifying what it is. Each entity has fields for the things you'd expect: position, view angles, health, team, current weapon, animation state. There's usually a single pointer somewhere to "the local player" — you, specifically, distinguished from everyone else because the client needs to know whose input to read and whose camera to render from.

For a first-person multiplayer game specifically, there's also a client-server split to know about: the server is authoritative over what actually happened, but the client predicts and interpolates so the game feels responsive despite network latency, and receives corrections when its prediction was wrong. That matters for cheating in one specific way — anything you fake purely on the client (a position, a hitbox) can be overruled or flagged by the server the moment it disagrees strongly enough with what the server itself is tracking. That's part of *why* certain cheats (aim precision, extra information) are harder for a server to catch than others (speed, teleportation): the server already has strong opinions about speed and position, and weaker ones about where you chose to aim.

# 3. How anti-cheat actually catches things

It's worth being concrete about this, because "anti-cheat" isn't one mechanism — it's several, usually layered:

- **Signature scanning.** The anti-cheat has a library of known cheats' code, byte patterns, or file hashes, and periodically scans the game's own process memory and loaded modules for a match. This catches popular, unmodified public cheats extremely well and custom, never-distributed ones almost not at all — there's nothing to match a signature against.
- **Integrity checking.** The anti-cheat hashes or otherwise verifies chunks of the game's own code and critical data structures at intervals, so that a hook (redirecting a function to run your code first) or a patched instruction shows up as a mismatch against what should be there.
- **Behavioral / statistical analysis.** Rather than looking for cheat code at all, this looks at what the player *did*: aim that snaps onto targets with inhuman consistency, reaction times below what any human demonstrably achieves, trigger timing that's suspiciously exact. This is the hardest category to build (a great player is also an outlier) and the hardest for a cheat to fully hide from, because it doesn't care how the advantage was obtained.
- **Delayed, server-side, wave-ban systems.** Valve's VAC is the canonical example: it doesn't act in real time. Suspected violations are queued and reviewed, then bans go out in batches, sometimes weeks later, deliberately so that a cheat developer can't immediately correlate "I changed X" with "I got banned" and patch around the specific trigger.
- **Kernel-mode monitoring.** The heaviest tier: rather than watching from inside the game's own process (usermode, same privilege level as the game and any external cheat reading it), the anti-cheat loads its own driver into the OS kernel, which can see essentially everything happening on the machine, at a privilege level no ordinary cheat process runs at. This is what modern anti-cheats like FACEIT AC, EAC, BattlEye, and Vanguard actually do, and it's significant enough that Part 2 of this series gives it its own section.

Most real anti-cheat deployments run several of these at once specifically because each one has different blind spots, and a technique that defeats one usually does nothing against the others.

# 4. Internal vs. external

Once you know what you want to read or change, there are two fundamentally different ways to reach it.

An **internal** cheat runs *inside* the game's own process — typically a DLL that gets loaded or injected into it. Once your code is running in that address space, reading and writing the game's memory is just reading and writing your own memory: no special permissions, no cross-process API calls, and — because you're in-process — you can hook the game's own functions directly (intercept its rendering calls to draw your own overlay through the game's own renderer, intercept its input handling, even call the game's own internal functions instead of reimplementing them). That last point matters more than it sounds: an internal aimbot can call the engine's own "set view angle" function directly, producing movement indistinguishable in principle from a real mouse turn, where an external one has to fake a mouse turn from outside and hope it looks the same. The cost is exposure: your code is now literally part of the process an anti-cheat is scanning, so it has to actively evade being noticed — unusual loaded modules, hooked functions, suspicious memory regions.

An **external** cheat runs as a *separate* process and reaches into the game's memory from outside, through operating-system APIs meant for exactly this (debuggers use the same calls). Nothing of yours ever executes inside the game's process, which sidesteps a whole category of in-process detection. The cost is capability: you can't hook the game's own rendering to draw through it, so anything you want to *see* — a radar, ESP boxes — needs its own overlay window layered on top, or has to be smuggled through a rendering mechanism the game already has (more on that trick specifically in Part 2). Anything you want to *do* beyond reading and writing raw values has to be done from outside too — simulating mouse and keyboard input, rather than calling the game's own "fire weapon" function directly.

Neither is strictly safer in general — it depends entirely on what the specific anti-cheat is watching for — but they're a genuinely different engineering problem, and it's worth deciding which one you're building before writing any code.

# 5. Injection: getting code into someone else's process

Injection is specifically the internal cheat's problem: how do you get your DLL loaded into a process you didn't start? The classic technique is almost boringly simple once you see it: `OpenProcess` to get a handle to the target, `VirtualAllocEx` to reserve some memory inside it, `WriteProcessMemory` to copy the path of your DLL into that memory, then `CreateRemoteThread` pointed at `LoadLibraryA` with that path as its argument. You've just made the target process call its own `LoadLibrary` on your DLL, from a thread you created, as if it had asked to load it itself — and Windows dutifully runs your `DllMain` inside their address space.

That specific sequence is also extremely well known, which is exactly why more evasive techniques exist. **Manual mapping** is the main one: instead of asking the loader to do the work, you do the loader's job yourself, from your injector process — parse your DLL's PE headers, allocate memory in the target for its sections, copy each section to its correct relative address, walk the import table and resolve every function your DLL calls against the target's already-loaded modules, apply base relocations if the memory didn't land at the DLL's preferred address, and finally call the entry point manually. It's considerably more code than the four-API-call version above, but the payoff is specific: because you never called `LoadLibrary`, your DLL never gets added to the target's module list, and any check that walks that list looking for unrecognized modules — which is a very common, very cheap first-line check — simply doesn't see it.

I'm not going to walk through manual mapping's implementation line by line here; the point isn't the recipe, it's understanding that "injection" is really just "make the target process do something it normally does — load and run some code — but on your behalf," and every detection method is really a way of noticing that something caused that to happen unexpectedly.

# 6. Reading and writing memory, for real

For an external cheat this is the entire toolkit, and it's a small one. `OpenProcess` gets you a `HANDLE` to the target, gated by exactly the access rights you request and the OS's own permission checks (you can't touch a process you don't have rights to, admin or not, if it's sufficiently protected). `ReadProcessMemory` and `WriteProcessMemory` then do exactly what they say, given that handle, an address, and a size — they're the same primitives a debugger uses to let you inspect a program you're stepping through.

The catch is that "an address" isn't something you know in advance. A game restarts with its memory laid out differently every time — address space layout randomization sees to that at the module level, and the loader picks its own base address for the executable and its DLLs on each run. What *is* stable, module-restart to module-restart, is the **offset** from a module's base address to a piece of data you care about, and the offset from the start of an object (a player entity, say) to one of its fields — because the compiler laid those out the same way every time it built that version of the game. So the actual workflow is: find the game process, find the base address of the module you care about right now (`client.dll` typically holds the interesting state in a Source-engine game), and add a previously-discovered offset to get today's real address. Those offsets have two flavors in most cheat codebases: ones relative to a module's base (often called **signatures**, because they're originally found by scanning the module's bytes for a unique instruction pattern) and ones relative to an object instance's own start (often called **netvars**, because in Source-engine games specifically they frequently match names the engine's own networking code uses internally, like `m_iHealth` or `m_vecOrigin`).

Very often you need to chase more than one offset in sequence — read a pointer at `base + offset_a` to get some manager object's address, then read again at `that_address + offset_b` to get the actual value you want. That's a **pointer chain**, and it's the norm rather than the exception: an entity list is usually a pointer to an array of pointers, so getting to a specific player's health is typically "read the entity-list pointer, read the pointer at that player's slot, then read the health field at a fixed offset from that entity's own address" — three memory operations for one number, every single refresh.

Every time the game updates, the compiler can lay memory out differently, and every offset can silently go stale — which is the single biggest maintenance burden in this whole exercise, and it's why offset lists are usually generated by a separate scanning tool rather than typed in by hand once and left alone.

# 7. The toolbox

A short list of what actually gets used, beyond a C or C++ compiler and the Windows API headers:

- **Cheat Engine** — a memory scanner built exactly for this: point it at a running process, search for a value (your health, say), change the value in-game, search again for what changed, and repeat until one address is left. It's the fastest way to find a single offset by hand without touching a disassembler.
- **A disassembler/decompiler** (IDA, Ghidra, x64dbg) — for finding *why* an address holds what it holds, following a function to understand a data structure's layout, or writing the byte-pattern signature that finds a given piece of code reliably across game versions.
- **A DLL injector** — for internal cheats, a small utility implementing the injection sequence from section 5 (or a manual mapper), so you're not rewriting it into every project.
- **An immediate-mode UI library, usually Dear ImGui** — the standard choice for cheat overlays specifically because it can be hooked into an existing renderer's present/frame call with very little glue code, which suits an internal cheat's "draw through the game's own renderer" approach well.
- **Public offset dumpers** — community-maintained tools and repositories that do the signature-scanning work in section 6 continuously, so individual projects can pull a fresh, current offset list instead of re-deriving it after every game patch.

# 8. Where people actually learn this

Almost none of the above is secret, and almost none of it was worked out from scratch by me — it's a small, well-documented field with an active community and a genuine on-ramp for beginners:

- **[UnknownCheats](https://www.unknowncheats.me)** is the oldest and largest forum for this specifically. It's organized by game, with entire sub-forums of source-available cheats, offset dumps kept current by the community, and long-running technical threads on injection, hooking, and anti-cheat internals — genuinely the first place to search when a specific game or technique is stuck.
- **[Guided Hacking](https://guidedhacking.com)** is the more structured alternative: a forum plus a large library of tutorials (many free, some paid) that walk through exactly the fundamentals in this post — injection, pattern scanning, hooking — as sequential lessons rather than scattered threads, which makes it a better starting point if you've never done any of this before.
- **Nick Cano's *Game Hacking: Developing Autonomous Bots for Online Games*** (No Starch Press, 2016) [1] is the closest thing this field has to a textbook — it covers pattern scanning, code injection, DLL and code caves, and building an actual bot end to end, in book form rather than forum-thread form.

That last community reference is exactly where khiar-ware's own offset supply comes from, which is where the next post picks up: khiar-ware doesn't do its own signature scanning at all — it pulls a community-maintained offset dump on demand, and the interesting engineering is entirely in what it does with those offsets afterward.

# References

1. N. Cano. *Game Hacking: Developing Autonomous Bots for Online Games.* No Starch Press, 2016. [nostarch.com/gamehacking](https://nostarch.com/gamehacking)
2. Microsoft. ReadProcessMemory function. Microsoft Learn. [learn.microsoft.com/.../nf-memoryapi-readprocessmemory](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-readprocessmemory)
3. Microsoft. WriteProcessMemory function. Microsoft Learn. [learn.microsoft.com/.../nf-memoryapi-writeprocessmemory](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-writeprocessmemory)
4. Microsoft. CreateRemoteThread function. Microsoft Learn. [learn.microsoft.com/.../nf-processthreadsapi-createremotethread](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createremotethread)
5. Microsoft. LoadLibraryA function. Microsoft Learn. [learn.microsoft.com/.../nf-libloaderapi-loadlibrarya](https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibrarya)
6. UnknownCheats. [unknowncheats.me](https://www.unknowncheats.me)
7. Guided Hacking. [guidedhacking.com](https://guidedhacking.com)
