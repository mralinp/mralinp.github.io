---
layout: post
title: "Smart Card: Building Your First Applet with jCardSim"
author: "Ali N. Parizi"
img: "/assets/images/posts/blog/smart-card/title-dev.jpg"
date:   2026-09-04 07:50:01 +0330
categories: blog embeded smart-card
brief: "In the previous post we introduced smart cards and Java Card applets. Now it's time to get our hands dirty: setting up a Java Card project, simulating a real card with jCardSim, and writing our first applet, one that generates a key pair on the card and signs a challenge with it, the same core trick behind a FIDO/U2F security key."
---

# 0. Recap and the goal

In the [previous post]({{ "/blog/embeded/smart-card/2023/07/19/smart-card-intro.html" | relative_url }}) we talked about what smart cards are, how they work, and what an applet is in the Java Card world: a small, isolated program that runs inside the card's secure microprocessor and talks to the outside world through APDU commands.

That post ended with a promise: it's time to actually write one.

But I also want to be upfront about *why* I'm going down this path. The real target of this series is not "print hello world on a card." It's building a hardware security token, a smart card that can act as a **FIDO/U2F security key**. FIDO keys (like a YubiKey) are, under the hood, exactly this kind of device: a secure element running an applet that generates and stores key pairs, signs authentication challenges, and never lets the private key leave the chip. Java Card gives us the same building blocks: isolated applet execution, on-card cryptography, and a well-defined command/response protocol over APDUs.

Before we touch real hardware, cryptography, or the FIDO protocol though, we need to get comfortable with the basics: writing an applet, installing it, and exchanging APDUs with it. And we're going to do all of that **without a physical smart card or reader**, using a simulator called jCardSim.

# 1. Why simulate the card first

Real Java Card smart cards are cheap, but the development loop around them is not fun:

- You need a physical card, a compatible reader, and often a card that accepts your own applets (many bank/SIM cards are locked down).
- Every test cycle means: compile, convert to a `.cap` file, load it onto the card over a reader, run it, and if something's wrong, erase and repeat.
- Debugging is close to impossible. There's no breakpoint, no stack trace, just a status word telling you something went wrong.

That's a painful way to learn a new, restricted dialect of Java. This is exactly the gap **jCardSim** fills.

# 2. What is jCardSim

[jCardSim](https://github.com/licel/jcardsim) is an open source, pure-Java simulator for the Java Card runtime environment (JCRE). It implements enough of the Java Card API (`javacard.framework`, `javacard.security`, and friends) to load and run real applet code, the same `.class` files you'd eventually convert into a `.cap` file for a physical card, entirely inside a JVM on your laptop.

A few things that make it genuinely useful, not just a toy:

- **No hardware required.** You write an applet, install it into a simulated card object, and start sending it APDUs immediately.
- **It speaks the same language as `javax.smartcardio`.** jCardSim exposes a `CardSimulator` that you drive with the standard Java smart card I/O API (`CommandAPDU` / `ResponseAPDU`), the exact same classes you'd use to talk to a real card through a PC/SC reader. That means code you write against the simulator is structurally the same code you'd use against real hardware later.
- **It's fast enough for real unit testing.** We used it on real projects to write automated tests for applet logic (state machines, PIN verification, data encoding) in a normal CI pipeline, no reader, no lab, no flaky USB drivers.
- **It's a stepping stone, not a replacement.** The simulator doesn't emulate every timing quirk or memory constraint of a real chip, so the last mile of testing on actual hardware still matters. But it removes 90% of the friction from the early development loop.

For prototyping something like a FIDO applet, where you'll be iterating a lot on command parsing, state handling, and cryptographic flows, this is exactly the tool you want before you ever plug in a real card.

# 3. Setting up the project

A minimal Java Card + jCardSim project needs two sets of dependencies:

1. The **Java Card Classic API** (`javacard.framework`, etc.), just to compile against the applet interfaces. jCardSim ships an implementation of these on the classpath, so for a simulation-only project you don't need Oracle/Global Platform's official Java Card Development Kit at all.
2. **jCardSim** itself, to get `CardSimulator`, `AIDUtil`, and the simulated runtime classes.

With Maven, a `pom.xml` for a small sandbox project looks roughly like this:

```xml
<dependencies>
    <dependency>
        <groupId>com.licel</groupId>
        <artifactId>jcardsim</artifactId>
        <version>3.0.5-SNAPSHOT</version>
    </dependency>
</dependencies>
```

The exact coordinates and available versions have moved around over the years (the project isn't always on Maven Central), so if that dependency doesn't resolve, grab the jar directly from the [jCardSim GitHub releases](https://github.com/licel/jcardsim) or build it from source with the repo's own instructions, and install it into your local `.m2` repository. Either way, once the jar is on your classpath you're ready to write the applet itself.

# 4. Our first applet: generate a key pair and sign a challenge

A "Hello World" applet wouldn't really tell us anything about the road ahead, so let's skip straight to the thing a security key actually does: **generate a key pair on the card, keep the private key locked inside it forever, and sign whatever challenge it's asked to sign.** That's the entire trust model behind FIDO/U2F, register a public key once, then prove possession of the matching private key on every login, and it's a handful of calls into `javacard.security`.

Every Java Card applet has the same shape: an `install()` factory method the JCRE calls when the applet is loaded, and a `process()` method the JCRE calls for every APDU sent to it. Ours generates its key pair once, at install time, and exposes two instructions: one to read back the public key, one to sign a challenge with the private key.

```java
package com.mralinp.simplesign;

import javacard.framework.APDU;
import javacard.framework.Applet;
import javacard.framework.ISO7816;
import javacard.framework.ISOException;
import javacard.security.KeyBuilder;
import javacard.security.KeyPair;
import javacard.security.RSAPublicKey;
import javacard.security.Signature;

public class SimpleSignApplet extends Applet {

    private static final byte INS_SIGN = (byte) 0x01;
    private static final byte INS_GET_PUBLIC_KEY = (byte) 0x02;

    private final KeyPair keyPair;
    private final Signature signer;

    protected SimpleSignApplet() {
        // Generate a fresh 1024-bit RSA key pair the moment the applet is installed.
        // The private key never leaves this object, and never leaves the card.
        keyPair = new KeyPair(KeyPair.ALG_RSA_CRT, KeyBuilder.LENGTH_RSA_1024);
        keyPair.genKeyPair();

        signer = Signature.getInstance(Signature.ALG_RSA_SHA_PKCS1, false);
        register();
    }

    public static void install(byte[] bArray, short bOffset, byte bLength) {
        new SimpleSignApplet();
    }

    @Override
    public void process(APDU apdu) {
        if (selectingApplet()) {
            return;
        }

        byte[] buffer = apdu.getBuffer();
        byte instruction = buffer[ISO7816.OFFSET_INS];

        switch (instruction) {
            case INS_GET_PUBLIC_KEY:
                sendPublicKey(apdu);
                break;
            case INS_SIGN:
                signChallenge(apdu);
                break;
            default:
                ISOException.throwIt(ISO7816.SW_INS_NOT_SUPPORTED);
        }
    }

    private void sendPublicKey(APDU apdu) {
        byte[] buffer = apdu.getBuffer();
        RSAPublicKey publicKey = (RSAPublicKey) keyPair.getPublic();

        // buffer[0] holds the exponent length, so the caller knows where the modulus starts.
        short exponentLength = publicKey.getExponent(buffer, (short) 1);
        buffer[0] = (byte) exponentLength;
        short modulusLength = publicKey.getModulus(buffer, (short) (1 + exponentLength));

        apdu.setOutgoingAndSend((short) 0, (short) (1 + exponentLength + modulusLength));
    }

    private void signChallenge(APDU apdu) {
        byte[] buffer = apdu.getBuffer();
        apdu.setIncomingAndReceive();
        short challengeLength = apdu.getIncomingLength();

        signer.init(keyPair.getPrivate(), Signature.MODE_SIGN);
        short signatureLength = signer.sign(buffer, ISO7816.OFFSET_CDATA, challengeLength, buffer, (short) 0);

        apdu.setOutgoingAndSend((short) 0, signatureLength);
    }
}
```

A couple of things worth calling out for anyone new to this API, since it looks like Java but behaves like embedded C in places:

- `install()` is a **static factory**, not a constructor you call yourself. The JCRE invokes it when the applet is loaded onto the card, and it's responsible for constructing the applet instance and calling `register()` so the card knows this applet exists and can be selected. Key generation happens right there, once, up front, exactly the way a real token generates its device key the moment it's provisioned.
- `process()` is called for **every single APDU**, including the `SELECT` command used to activate the applet before anything else can happen. That's why the very first thing we do is check `selectingApplet()` and return early, we don't want to fall into our instruction switch for a `SELECT`.
- Everything happens through a **shared APDU buffer** (`apdu.getBuffer()`). There's no heap of objects being passed around, incoming challenge bytes are read straight out of the buffer and the signature is written straight back into it. This is a direct consequence of how little RAM a real smart card chip has.
- The `RSAPrivateCrtKey` half of `keyPair` is never read, copied, or sent anywhere in this code. It only ever gets handed to `signer.init(...)`, which uses it internally to produce a signature. That's the whole point: the card proves it holds the key without ever exposing it.

We're using RSA here rather than the ECDSA over P-256 that real FIDO devices use, mainly because RSA key generation needs nothing beyond a modulus length, while EC key generation on Java Card normally requires setting explicit curve domain parameters first. It keeps this first example short. When we build the real FIDO applet, we'll switch `KeyPair.ALG_RSA_CRT` for `KeyPair.ALG_EC_FP` and set up the P-256 domain parameters properly.

# 5. Running it in jCardSim

Now the payoff: let's install this applet into a simulated card, ask it for its public key, hand it a challenge to sign, and verify the signature ourselves, all without a physical card or reader.

```java
package com.mralinp.simplesign;

import com.licel.jcardsim.smartcardio.CardSimulator;
import com.licel.jcardsim.utils.AIDUtil;
import javacard.framework.AID;

import javax.smartcardio.CommandAPDU;
import javax.smartcardio.ResponseAPDU;
import java.math.BigInteger;
import java.security.KeyFactory;
import java.security.PublicKey;
import java.security.Signature;
import java.security.spec.RSAPublicKeySpec;
import java.util.Arrays;

public class SimpleSignTest {

    public static void main(String[] args) throws Exception {
        // 1. Create a simulated card and install our applet on it
        CardSimulator simulator = new CardSimulator();
        AID appletAID = AIDUtil.create("A000000003000002");
        simulator.installApplet(appletAID, SimpleSignApplet.class);
        simulator.selectApplet(appletAID);

        // 2. Ask the card for the public half of the key it generated at install time
        ResponseAPDU pubKeyResponse = simulator.transmitCommand(new CommandAPDU(0x00, 0x02, 0x00, 0x00));
        byte[] pubKeyBytes = pubKeyResponse.getData();

        int exponentLength = pubKeyBytes[0] & 0xFF;
        BigInteger exponent = new BigInteger(1, Arrays.copyOfRange(pubKeyBytes, 1, 1 + exponentLength));
        BigInteger modulus = new BigInteger(1, Arrays.copyOfRange(pubKeyBytes, 1 + exponentLength, pubKeyBytes.length));

        PublicKey publicKey = KeyFactory.getInstance("RSA")
                .generatePublic(new RSAPublicKeySpec(modulus, exponent));

        // 3. Send a "login challenge" and have the card sign it with its private key
        byte[] challenge = "login-challenge-42".getBytes();
        ResponseAPDU signResponse = simulator.transmitCommand(new CommandAPDU(0x00, 0x01, 0x00, 0x00, challenge));
        byte[] signature = signResponse.getData();

        // 4. Verify the signature ourselves, exactly like a server would on the other end
        Signature verifier = Signature.getInstance("SHA1withRSA");
        verifier.initVerify(publicKey);
        verifier.update(challenge);

        System.out.println("Signature valid: " + verifier.verify(signature));
    }
}
```

Running it prints:

```text
Signature valid: true
```

Nothing about steps 3 and 4 talked to the card, the private key stayed inside the simulated chip the entire time. We only ever handed the card a challenge and got a signature back, then verified that signature ourselves using nothing but the public key it gave us in step 2. That round trip, hand the token a challenge, get back a signature you can verify against a previously-registered public key, is the actual core of how a FIDO/U2F authentication works.

# 6. Where this is going

This applet is already doing the two things a real security key needs to do: keep a private key that never leaves the chip, and prove possession of it by signing on demand. What's missing is the FIDO/U2F protocol wrapped around that behaviour, a `REGISTER` command that returns a properly formatted attestation, an `AUTHENTICATE` command that tracks and increments a usage counter, ECDSA over the P-256 curve instead of RSA, and key handles instead of a single fixed key pair.

In the next post, we'll build on this same jCardSim setup and start implementing the actual FIDO U2F command set on top of it, one instruction at a time, before ever touching a real card. Stay tuned.
