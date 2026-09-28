---
layout: post
title:  "From Keypad Tones to Embedded Linux: Two Years of Embedded R&D"
author: "Ali Naderi"
img: "/assets/images/posts/projects/embedded-rnd/cover.jpg"  # TODO: add a photo of the board(s)
date:   2026-09-28 12:00:00 +0330  # TODO: set the publish date
categories:  project embedded hardware fpga linux
brief: "A phone line you can talk to with the keypad, and a Linux board that watches an analog camera for motion: what I built during two years of embedded R&D at Sico Systems, and what it took."
---
From 2017 to 2019 I worked in R&D at Sico Systems Intelligizer in Shiraz, first as an intern, then as a junior embedded engineer. The work was FPGAs, microcontrollers, and hardware that had to survive the real world. Two builds from that time are worth writing down: a device you control by phoning it and pressing keys, and a small Linux system-on-board that takes an analog camera feed and detects motion on the board itself.

> **TODO:** one or two sentences on what Sico Systems made, and why these two projects existed (a product? a client request? internal research?).

# 1. DTMF: the tones behind the keypad

Every key on a phone keypad sends two tones at once, one low and one high. That scheme is **DTMF** (dual-tone multi-frequency), and it's why "press 1 for sales" works over any phone line: the tones are in the voice band, so they travel wherever a voice can.

|           | 1209 Hz | 1336 Hz | 1477 Hz | 1633 Hz |
| --------- | :-----: | :-----: | :-----: | :-----: |
| **697 Hz** |    1    |    2    |    3    |    A    |
| **770 Hz** |    4    |    5    |    6    |    B    |
| **852 Hz** |    7    |    8    |    9    |    C    |
| **941 Hz** |    *    |    0    |    #    |    D    |

Pressing **5** sends 770 Hz and 1336 Hz together. A detector's job is to find exactly one tone from each group, confirm it lasts long enough to be a real key press and not speech that happens to contain those frequencies, and report which key it was.

## 1.1 The detector

The first step was a DTMF detector in hardware.

> **TODO:** How was it built? A dedicated decoder chip, analog filters, the FPGA, or a microcontroller doing the math (e.g. the Goertzel algorithm)? What did it output (a 4-bit code + "valid" strobe, a UART message)? A schematic or a photo of the prototype would go well here.

> **TODO:** The hard parts: false triggers from speech or noise, minimum tone duration, twist (level difference between the two tones), and how you tested it.

# 2. A phone number that takes orders

Then the detector got a phone line. With a **GSM/GPRS module**, the device had its own SIM and number. You call it, it answers, and the call becomes a menu:

- it **plays voice prompts** ("press 1 for …", "enter your access code"),
- it **reads the keys you press** through the DTMF detector,
- and it **acts on them**: runs a command, or checks a code before granting access.

No app, no internet, no screen: any phone, anywhere with GSM coverage, is the remote control.

> **TODO:** What did it control in practice (relays, a door, an alarm, machines)? Which GSM/GPRS module? How were the prompts stored and played (the module's audio, an MCU with a DAC, a voice chip)? What was the GPRS part used for, if anything?

## 2.1 Signal quality

The design used several chips, and getting them to work together over a phone call was a problem of its own: the audio path from the GSM module into the detector has to stay clean enough for the tones to be recognised reliably.

> **TODO:** What went wrong at first and how you fixed it: noise from the GSM module's transmit bursts (the classic 217 Hz "buzz"), levels and impedance between chips, grounding and layout, filtering. Before/after numbers or scope captures would make this section.

# 3. A Linux system-on-board

The next project was researching a **system-on-board** (SoB): a small board with a real CPU and memory that runs a full Linux, instead of a microcontroller running a single loop.

## 3.1 Choosing the chip

I reviewed several Chinese SoC chipsets and settled on the **Allwinner F1C100S**, the chip on the Lichee Pi Nano: a small ARM SoC with its memory in the same package, cheap, and able to run mainline-style embedded Linux.

> **TODO:** Which other chips were on the list (e.g. other Allwinner parts, Rockchip, Ingenic) and why the F1C100S won: price, memory in package, hand-solderable package, Linux support, video features?

## 3.2 The board

The board started from the Lichee Pi Nano's design and added what the project needed, most importantly an **analog video input**, so an ordinary analog (CCTV-style) camera could feed it directly.

> **TODO:** What else you added or changed. Bootloader, kernel and root filesystem (U-Boot, Buildroot/Yocto?), how the video input was wired and driven, and anything that took a week to debug.

## 3.3 Seeing motion on the board

With frames coming in, the board processed them itself. One of the filters was the **Laplacian**, a second-derivative operator that responds strongly where brightness changes sharply, which is to say at edges:

```text
 0  1  0
 1 -4  1
 0  1  0
```

Convolve a frame with that kernel and flat areas go to zero while edges light up. Compare the edge maps of consecutive frames and anything that moved stands out, while slow changes in overall lighting (which move every pixel together and have no edges of their own) mostly cancel.

> **TODO:** The exact pipeline (frame difference then Laplacian, or Laplacian then difference? thresholds? a minimum blob size?), the resolution and frame rate the F1C100S managed, and what happened when motion was detected.

# 4. What it taught me

> **TODO:** A short closing: what two years of hardware did for the way you write software (timing, resources, debugging without a debugger), and a photo of the boards side by side.

# References

1. ITU-T Recommendation Q.23, *Technical features of push-button telephone sets* (the DTMF frequencies).
2. Allwinner F1C100S datasheet. <!-- TODO: link -->
3. Lichee Pi Nano documentation. <!-- TODO: link -->
