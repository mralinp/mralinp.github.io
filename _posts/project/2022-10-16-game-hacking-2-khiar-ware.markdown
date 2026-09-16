---
layout: post
title:  "Game Hacking, Part 2: khiar-ware, an External CS:GO Cheat"
author: "Ali N. Parizi"
img: "/assets/images/posts/projects/khiar-ware/demo.png"
date:   2022-10-16 12:21:13 +0330
categories:  project security cracking game-hacking
brief: "A report on khiar-ware — an external CS:GO cheat: reading and writing csgo.exe's memory from a separate process, keeping offsets in sync with the game, and why kernel-mode anti-cheat is the technique's actual ceiling."
github: "https://github.com/mralinp/khiar-ware"
---
[khiar-ware](https://github.com/mralinp/khiar-ware) is a small Windows cheat for CS:GO I built to put [Part 1](/project/security/cracking/game-hacking/2022/10/09/game-hacking-1-fundamentals.html)'s concepts into working code: WallHack, TriggerBot, RadarHack, and AutoBhop, all from a single external C++ process. It's GPL-3.0, it's a few years old at this point, and its offsets are almost certainly stale against the current game — CS:GO itself was superseded by CS2 in 2023, and Source-engine `client.dll` offsets don't survive an engine migration. It stands as a working reference for the techniques, not a tool for touching a live match.

**Educational purposes only.** Using anything like this on a VAC-secured server risks a permanent ban, and cheating against real opponents is exactly the kind of thing that makes online games worse for everyone else. Testing shown here was done entirely offline, in CS:GO's own `aim_botz` bot-practice map, with the game launched in `-insecure` mode — which disables VAC and, as a direct consequence, also blocks you from joining any VAC-secured server at all. That's not a workaround; it's the point.

<p align="center">
    <img width="85%" src="/assets/images/posts/projects/khiar-ware/demo.png"/>
</p>
<p align="center"><em>khiar-ware's glow-based wallhack and radar running against bots in aim_botz, tested with VAC disabled via -insecure.</em></p>

# 1. Shape of the thing

Per Part 1's terms, khiar-ware is an **external** cheat: it's its own console executable, not a DLL injected into `csgo.exe`. `main()` prints a banner, sets up the four feature classes, and then runs a plain polling loop — call each feature's `refresh()`, check for a hotkey, sleep one millisecond, repeat:

```cpp
while (true) {
    glow.refresh();
    radar.refresh();
    bhop.refresh();
    trigger.refresh();
    int key = get_key();
    // INSERT toggles glow, HOME toggles the triggerbot, PAGE UP toggles bhop
    ...
    Sleep(1);
}
```

Every feature is independent, toggled at runtime, and reads/writes the target process through one shared object — everything funnels through a single `MemoryManager`. (There's also a fifth class in the repo, `ScriptBot`, that's never instantiated in `main()` at all — dead code from an earlier idea, left in place. It's a decent reminder that a project's actual feature set is what's wired up, not everything sitting in the source tree.)

# 2. Talking to csgo.exe

`MemoryManager` does exactly the two things Part 1 said an external cheat needs: find the target process, and give every feature a typed way to read and write it.

Finding the process and its modules is standard Windows tool-help API [2] — enumerate running processes for `csgo.exe`, `OpenProcess` [3] on it, then walk its loaded modules with `CreateToolhelp32Snapshot` [2] to find `client.dll` and `engine.dll` and record their base addresses. That base address is the "today's starting point" from Part 1's section 6 — every feature's memory access is that base plus a fixed offset.

Reading and writing themselves are two small template functions wrapping `ReadProcessMemory`/`WriteProcessMemory` [4, 5], so the rest of the codebase can write `mem->read<Vector>(address)` or `mem->write<int>(address, 6)` instead of juggling raw byte buffers everywhere:

```cpp
template<class type>
auto read(DWORD dwAddress) {
    type buff{};
    ReadProcessMemory(this->_hproc, (LPVOID)dwAddress, &buff, sizeof(type), NULL);
    return buff;
}

template<class type>
BOOL write(DWORD dwAddress, type ValueToWrite) {
    return WriteProcessMemory(this->_hproc, (LPVOID)dwAddress, &ValueToWrite, sizeof(type), NULL);
}
```

Every feature below is built entirely out of calls to these two functions at specific offsets. There's no hooking, no injected code, no rendering takeover — just reading and occasionally writing plain memory from outside the process.

# 3. Offsets that update themselves

`Offsets.h` holds two lists of named constants — `signitures` (module-relative addresses like `dwLocalPlayer`, `dwEntityList`, `dwGlowObjectManager`, `dwForceJump`) and `netvars` (offsets *within* a player entity, like `m_iHealth`, `m_vecOrigin`, `m_iTeamNum`, `m_iCrosshairId`, `m_bSpotted`) — around eighty names in total between the two.

None of them are hand-derived. `updater.py` regenerates the whole file from [hazedumper](https://github.com/frk1/hazedumper) [1], a community-maintained CS:GO offset dump that did the actual signature-scanning work Part 1 described:

```python
url = 'https://raw.githubusercontent.com/frk1/hazedumper/master/csgo.json'
r = requests.get(url, allow_redirects=True)
# ...parse csgo.json's "signatures" and "netvars" objects,
# write each one out as a `const DWORD` into Offsets.h
```

That's the practical answer to Part 1's maintenance problem: rather than re-deriving offsets by hand after every game patch, regenerate the header from an external dump and rebuild. It also means khiar-ware's own currency is entirely tied to hazedumper's — once a public dumper for a game goes unmaintained, so does everything built on top of it, which is a big part of why this particular project is stale today.

# 4. The four features

**SimpleGlow (WallHack).** The interesting choice here is that it doesn't draw anything at all. CS:GO's engine already has a glow-rendering system (used for the legitimate "outline teammates through walls" setting), backed by an array of glow structs — one per glowable entity, holding color, alpha, and a couple of render flags including `m_bRenderWhenOccluded`. Every refresh, `SimpleGlow` walks the entity list, and for every living, non-dormant entity on the opposing team, reads that entity's glow struct, sets its color from a red-to-green ramp based on current health, flips `m_bRenderWhenOccluded` on, and writes the whole struct back:

```cpp
gst = mem->read<Glow_Struct>(glowObj + (glowIdx * 0x38));
gst.G = health * 0.01f;
gst.R = health * -0.01f + 1.0f;
gst.m_bRenderWhenOccluded = true;
mem->write<Glow_Struct>(glowObj + (glowIdx * 0x38), gst);
```

One struct write per enemy per frame, and the game's own renderer does the rest — no overlay window, no drawing code of its own.

**TriggerBot.** While Alt is held, it reads the local player's `m_iCrosshairId` netvar — the engine's own record of which entity index is currently under the crosshair — and if that entity exists and isn't on the local player's team, fires: a short randomized delay, a `mouse_event(MOUSEEVENTF_LEFTDOWN)`, another short delay, then `LEFTUP`. Because it's external, it can't call the game's "fire" function directly the way an internal hook could — it has to go through the same OS-level input simulation any input-automation tool uses, which is also why it needs the crosshair check at all: it has no way to fire *at* a target, only to fire *when* the game says one is already under the crosshair.

**AutoBhop.** The smallest feature by far: while Space is held and the local player's `m_fFlags` netvar has the on-ground bit set, it writes a nonzero value to `dwForceJump` — a single engine-global address the game's own movement code checks each tick to decide whether to force a jump regardless of normal input timing. No aiming, no entity list, one conditional memory write.

**Radar.** This one turns out to use the exact same trick as SimpleGlow, and I described it wrong to myself when I first sketched this post — it doesn't draw a custom overlay at all. CS:GO already tracks a per-entity `m_bSpotted` netvar, which the game's own built-in radar/minimap UI uses to decide who to show: normally it's only true for an enemy your team has actually spotted. `Radar::refresh()` just walks up to 32 entity slots and force-writes `m_bSpotted = true` on every non-dormant one, every frame:

```cpp
for (int i = 0; i < 32; i++) {
    DWORD entity = getEntity(i);
    if (entity != NULL && isDormant(entity) == false) {
        mem->write<bool>(entity + netvars::m_bSpotted, true);
    }
}
```

One boolean write per entity, and the game's existing minimap shows everyone. Between this and SimpleGlow, two of khiar-ware's four features work by flipping a flag the engine already checks for a legitimate purpose, rather than building any rendering of their own — which in hindsight is the most reusable idea in the whole project: before writing an overlay, check whether the game already has a rendering path you can just redirect.

# 5. Testing without risking the account

The setup section of the repo's README is worth taking seriously on its own, because it's the difference between "testing" and "guaranteed ban." CS:GO's Steam launch options accept `-insecure`, which explicitly disables VAC for that session — and, as a direct trade-off, blocks the client from joining any VAC-secured server at all. That combination is exactly what makes it useful for this kind of work: there's no server to accidentally cheat against, and no anti-cheat process running to trip. The screenshot above is `aim_botz`, a community-made offline practice map full of stationary and moving bots, played entirely locally — a reasonable place to see whether a memory offset is even correct before it matters at all whether anyone's watching.

# 6. Ring 0: the ceiling this technique runs into

Everything above — and everything in Part 1 — lives entirely in what Part 1 called usermode: khiar-ware, `csgo.exe`, and even a hypothetical kernel-unaware anti-cheat process all run at the CPU's **Ring 3**, the least-privileged of the protection rings x86 defines (Rings 1 and 2 exist in the architecture but neither Windows nor Linux actually use them; in practice it's a two-tier system, Ring 3 for applications and **Ring 0** for the OS kernel and its drivers). The asymmetry between the two rings is total: Ring 0 code can inspect and modify *any* process's memory, including another Ring 0 driver's, with no permission check standing in its way, because it *is* the thing that would normally enforce that check. Ring 3 code — no matter how privileged the user account running it — has no equivalent visibility into Ring 0 at all.

That asymmetry is the entire reason kernel-mode anti-cheat exists, and it's worth being honest about what it means for everything in this post: a driver running at Ring 0 can directly watch every `OpenProcess` call made against the protected game, from any process, with any requested access rights — which is precisely the API khiar-ware's `MemoryManager` calls to get started. It doesn't need to catch khiar-ware's specific offsets or behavior; the act of opening a handle to `csgo.exe` with memory-read access from an unrelated process is itself the tell. FACEIT's own documentation describes exactly this shape: their Anti-Cheat is a usermode app plus a kernel-mode driver loaded at boot, and more recently a Secure Boot and TPM 2.0 requirement on top of that [5]. EAC, BattlEye, and Riot's Vanguard are built the same way, for the same reason [6, 7]. None of khiar-ware's techniques would survive first contact with any of them, and that's true independent of whether its offsets happen to be current.

So — purely as an idea, because "how do you actually reach Ring 0" is a real technique with real consequences and not this post's subject — what does having Ring 0 yourself change? A few concrete things, at a conceptual level:

- **You stop using the APIs the anti-cheat is watching.** Reading and writing memory from kernel mode uses different, kernel-internal mechanisms (for instance `MmCopyVirtualMemory`) rather than `ReadProcessMemory`/`OpenProcess`, so the specific usermode calls a Ring 3-only detector watches for simply never happen.
- **You can hide your own existence from anything enumerating the normal way.** This is usually called DKOM — Direct Kernel Object Manipulation — and it means editing the kernel's own bookkeeping structures directly: for instance, unlinking your driver or process from the linked list the kernel walks to answer "what's running right now," so a query that lists processes the standard way never sees you, even though you're still executing.
- **You're now visible to, and in a fair fight with, anything else at Ring 0** — which very much includes the anti-cheat's own driver, if it's also kernel-mode. Reaching Ring 0 isn't reaching a blind spot; it's reaching the same floor the anti-cheat is standing on, which is exactly why kernel anti-cheats spend so much of their effort watching *that* layer specifically rather than assuming it's safe by default [6].

There are, publicly and broadly, two routes into Ring 0 on modern Windows, and both are extensively written about rather than secret: writing your own kernel driver and getting it loaded, which is deliberately hard, because Windows requires kernel-mode drivers to carry a valid digital signature before the loader will run them at all [8]; or **BYOVD** — Bring Your Own Vulnerable Driver — where instead of writing new kernel code, you load an *already*-signed, legitimately trusted third-party driver that happens to have a bug (commonly, an IOCTL handler that will read or write arbitrary physical or kernel memory on request, with no check on who's asking), and drive that existing bug from usermode to get de-facto kernel read/write without ever authoring or signing a driver of your own [9, 10]. This is exactly serious enough a problem that there's a dedicated public catalog, [LOLDrivers](https://www.loldrivers.io) [11], tracking known-vulnerable signed drivers specifically so defenders — including anti-cheat vendors — can detect and block them; the arms race here runs in both directions, in public, on both sides.

The takeaways, stripped of everything else in this section:

- Ring 3 is always visible to Ring 0; the reverse is never true. That single fact is the entire justification for kernel-mode anti-cheat existing at all.
- Anything crossing the user/kernel boundary through a documented API — `OpenProcess`, `ReadProcessMemory`, `CreateRemoteThread` — is exactly what a kernel-mode anti-cheat driver is positioned to observe directly, regardless of how well the usermode side is written.
- Reaching Ring 0 yourself, by any route, doesn't grant invisibility — it grants a fair fight, on the same floor the anti-cheat already occupies.
- None of these techniques are secret on either side. BYOVD, DKOM, and driver-signature enforcement are all publicly documented specifically *because* defenders need to know them too — which is a decent one-sentence summary of why this whole field keeps moving.

# 7. What building it actually taught

The instructive part, in hindsight, wasn't any single feature — it was how differently each one reached the same goal, and how two of the four turned out to need no rendering code at all once I looked closely enough at what the engine already tracked. None of it would survive a client that re-randomized its own internal layout on every build, encrypted these structures, or moved this state server-side entirely — and, as section 6 lays out, modern anti-cheat has, in various ways, done exactly that, one ring up from everything khiar-ware does. It's a genuinely well-matched arms race, and building even a small, dated external cheat like this one is a reasonable way to feel out why.

# References

1. F. (frk1). hazedumper. GitHub. [github.com/frk1/hazedumper](https://github.com/frk1/hazedumper)
2. Microsoft. CreateToolhelp32Snapshot function. Microsoft Learn. [learn.microsoft.com/.../nf-tlhelp32-createtoolhelp32snapshot](https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/nf-tlhelp32-createtoolhelp32snapshot)
3. Microsoft. OpenProcess function. Microsoft Learn. [learn.microsoft.com/.../nf-processthreadsapi-openprocess](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-openprocess)
4. Microsoft. ReadProcessMemory function. Microsoft Learn. [learn.microsoft.com/.../nf-memoryapi-readprocessmemory](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-readprocessmemory)
5. FACEIT. What is FACEIT Anti-cheat and how does it work? FACEIT Support. [support.faceit.com](https://support.faceit.com/hc/en-us/articles/9394666828188-What-is-FACEIT-Anti-cheat-and-how-does-it-work)
6. C. Dorner, L. D. Klausner. If It Looks Like a Rootkit and Deceives Like a Rootkit: A Critical Examination of Kernel-Level Anti-Cheat Systems. arXiv:2408.00500, 2024. [arxiv.org/abs/2408.00500](https://arxiv.org/abs/2408.00500)
7. s4dbrd. How Kernel Anti-Cheats Work: A Deep Dive into Modern Game Protection. 2026. [s4dbrd.github.io](https://s4dbrd.github.io/posts/how-kernel-anti-cheats-work/)
8. Microsoft. Driver Signing With Digital Signatures. Microsoft Learn. [learn.microsoft.com/.../driver-signing](https://learn.microsoft.com/en-us/windows-hardware/drivers/install/driver-signing)
9. Bitdefender. What is Bring Your Own Vulnerable Driver (BYOVD). Bitdefender TechZone. [techzone.bitdefender.com](https://techzone.bitdefender.com/en/tech-explainers/what-is-bring-your-own-vulnerable-driver--byovd-.html)
10. Picus Security. What Is a BYOVD Attack? Bring Your Own Vulnerable Driver, Explained. [picussecurity.com](https://www.picussecurity.com/resource/blog/what-are-bring-your-own-vulnerable-driver-byovd-attacks)
11. LOLDrivers. [loldrivers.io](https://www.loldrivers.io)
