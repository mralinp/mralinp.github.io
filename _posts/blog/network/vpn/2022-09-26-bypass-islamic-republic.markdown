---
layout:     post
title:      "How to bypass the 'Islamic Republic' internet filtering?"
author:     "Ali Naderi"
img:        "/assets/images/posts/blog/vpn-setup/title.png"
cover-img: "/assets/images/posts/blog/vpn-setup/azadi.jpg"
date:       2022-09-26  17:50:22 +0330
categories: blog network vpn 
brief:      "An introduction to virtual private networks(VPN), tunnels, proxies, and a straightforward walkthrough for running a WireGuard VPN server."
---

# 1. Intro

Every so often, we hear stories from around the world of governments clashing with their own people — over unjust laws, opaque decisions, a struggling economy, financial corruption, and no shortage of other reasons. One of their favorite ways to regain control is to cut people off from the world, forcing ISPs and top-tier providers to shut down internet connections. Some of these countries are China, North Korea, and the Islamic Republic of Iran. People living in these countries often do their jobs through the internet, so their livelihood is directly tied to staying connected to online services. In this situation, some of the jobs most at risk are developers, online shop owners, reporters, and many more. As a developer, I can't live without the internet — my professional life is intertwined with this technology. So I have to solve this problem myself, because I refuse to accept the government's politics on this. Let's begin.

# 2. What is the solution
How can we access the outside world? We need a machine that's already connected to the internet, and from there, we can reach the rest of the world. In these situations, governments generally won't disconnect data centers from the internet, since doing so puts their own servers and companies at serious risk, especially from a security standpoint. So we can conclude that data centers inside the country stay connected to the internet and can still reach the outside world.

Even if we could connect to the world through a machine in a local data center, there's still a problem for our freedom: the **cruel sanctions imposed by the United States**, which prevent these people from accessing services and content that's freely available to everyone else in the world. To tackle this problem, we need a second machine, one that's in another country and accessible through the internet. We route our packets through that second machine to be fully free. To do so, we need to build a **virtual private network** (VPN).

## 2.1 Virtual Private Network (VPN)
A virtual private network extends a private network across a public network and enables users to send and receive data across shared or public networks as if their computing devices were directly connected to the private network. The benefits of a VPN include increases in functionality, security, and management of the private network. 

<div align=center>
    <img src="/assets/images/posts/blog/vpn-setup/vpn-schema.png"/>
</div>
<br>

It provides access to resources that are inaccessible on the public network and is typically used for remote workers. Encryption is common, although not an inherent part of a VPN connection. A VPN is created by establishing a virtual point-to-point connection through the use of dedicated circuits or with tunneling protocols over existing networks. A VPN available from the public Internet can provide some of the benefits of a wide area network. From a user perspective, the resources available within the private network can be accessed remotely ([Wikipedia](https://en.wikipedia.org/wiki/Virtual_private_network)).

# 3. Running a WireGuard VPN server on Ubuntu 20.04 (LTS)
WireGuard is a communication protocol and free and open-source software that implements encrypted virtual private networks, and was designed with the goals of ease of use, high speed performance, and low attack surface. It aims for better performance and more power than IPsec and OpenVPN, two common tunneling protocols ([Wikipedia](https://en.wikipedia.org/wiki/WireGuard)). 

<div align=center>
    <img width="50%" src="/assets/images/posts/blog/vpn-setup/wg-logo.png"/>
</div>

For this tutorial, I chose WireGuard as the VPN protocol for this article. Installing and configuring a WireGuard VPN server is quite simple and easy to understand, even for those who aren't familiar with some Linux networking concepts, especially in comparison with other protocols such as OpenVPN.

To get started, you need a Virtual Machine(VM) accessible through the internet via SSH in outside world. Let's assume my VM IP address is `77.222.67.140`, I can connect to it using SSH as below:

```console
$ ssh root@77.222.67.140
```

After running the command above, it asks you to type "yes" if you trust this host, just type "`yes`" and don't ask why. Then you have to enter the VM password.
The first thing you do after connecting to the virtual machine is updating operating system packages to the latest available version using `Aptitude Package Manager`(apt):

```console
$ apt update --yes
$ apt upgrade --yes
```

> Note: It’s recommended that you reboot and reconnect to the VM after upgrading its packages.

Now we have to install `wireguard` and `wireguard-tools` using apt:

```console
$ apt install wireguard --yes
$ apt install wireguard-tools --yes
```

This should install the WireGuard kernel module and the necessary tools for running our VPN server. If you would like to route your WireGuard Peer’s Internet traffic through the WireGuard Server then you will need to configure IP forwarding. To configure forwarding, open the `/etc/sysctl.conf` file using vim or your preferred editor:

```console
$ vim /etc/sysctl.conf 
```

Then you have to look for a line containing `net.ipv4.ip_forward=1` and uncomment that line (remove leading `#`) Or, you can just add this text at the end of `sysctl.conf` file. If you are using IPv6 with WireGuard, uncomment/add line `net.ipv6.conf.all.forwarding=1`. If you are using both IPv4 and IPv6, ensure that you include both lines. Save and close the file when you are finished (if you were using vim press `ESC` then type `wq` and press `enter`). To read the file and load the new values for your current terminal session, run:

```console
$ sysctl -p
```
If you did it right, you have to see an output as below:
```output
net.ipv6.conf.all.forwarding = 1
net.ipv4.ip_forward = 1
```

Now your WireGuard Server will be able to forward incoming traffic from the virtual VPN ethernet device to others on the server, and from there to the public Internet. Using this configuration will allow you to route all web traffic from your WireGuard Peer via your server’s IP address, and **your client’s public IP address will be effectively hidden**.

However, before traffic can be routed via your server correctly, you will need to configure some firewall rules. These rules will ensure that traffic to and from your WireGuard Server and Peers flows properly.

## 3.1 WireGuard UI
[**WireGuard UI**](https://github.com/ngoduykhanh/wireguard-ui) is a web-based config generator for the WireGuard server. If you've seen the DigitalOcean tutorial for running a WireGuard server on Ubuntu 20.04, from which I borrowed some parts of this tutorial (or other popular tutorials), you'll know they use the command line to generate client configurations, a process called **adding peers**, for the WireGuard server. Using the command line interface and `wg` directly is quite hard to manage if you're building a network for your co-workers, family, or friends. Wireguard-ui is a web-based interface for generating and managing client profiles, and it's written in Go, which means that if you use the binary files available on their [**releases page**](https://github.com/ngoduykhanh/wireguard-ui/releases), you don't have to worry about building the application, and it should work without any problems.

To start using wireguard-ui, you need to download the binary files first:

```console
$ wget https://github.com/ngoduykhanh/wireguard-ui/releases/download/v0.3.7/wireguard-ui-v0.3.7-linux-amd64.tar.gz
```

Then unzip the downloaded file:

```console
$ tar -xvf wireguard-ui-v0.3.7-linux-amd64.tar.gz
```

Before running wireguard-ui, you have to open port 5000 on your VM which is the default port of wireguard-ui:

```
$ ufw allow 5000 # Open 5000 port
$ ufw disable # stop the firewall
$ ufw enable  # start the firewall
```

Now, run wireguard-ui using:

```console
$ ./wireguard-ui
```
> Note: If the above command failed, make sure that you gave the run access to that binary file using `$ chmod +x wireguard-ui`

After running wireguard-ui you can open your browser and type `YOUR_MACHINE_ADDRESS:5000` at the address bar and start using wireguard-ui. The default username and passwords for wireguard-ui are:

```text
username: admin
password: admin
```
Login to the panel and make as many client as you want, then download the configuration files or save the qr code for each client. After you finished your job, press `apply config` and go back to your vm, terminate the wireguard-ui by pressing `ctrl + c`. 

<div align=center>
    <img src="/assets/images/posts/blog/vpn-setup/wireguard-ui.png"/>
</div>

<br>

> Note: Before making user clients, I recommend you to first change the server's default port (then press `apply config`) and then begin creating profiles.

Wireguard-ui should generate a configuration file and place it inside `/etc/wireguard/wg0.conf`. After terminating wireguard-ui, no further configurations are needed for adding clients and you can give the downloaded config files to your clients. To run the server, there are two more steps to go: configuring the server's firewall, and running the WireGuard server as a background service, both covered in the next sections.


## 3.2 Configuring the WireGuard Server’s Firewall

In this section you will edit the WireGuard Server’s configuration to add firewall rules that will ensure traffic to and from the server and clients is routed correctly. As with the previous section, skip this step if you are only using your WireGuard VPN for a machine to machine connection to access resources that are restricted to your VPN.

To allow WireGuard VPN traffic through the Server’s firewall, you’ll need to enable masquerading, which is an iptables concept that provides on-the-fly dynamic network address translation (NAT) to correctly route client connections.

First find the public network interface of your WireGuard Server using the ip route sub-command:

```console
$ ip route list default
```

The public interface is the string found within this command’s output that follows the word “dev”. For example, this result shows the interface named eth0, which is highlighted below:

```output
default via 77.222.67.140 dev eth0 proto static
```
Note your device’s name since you will add it to the iptables rules in the next step.
To add firewall rules to your WireGuard Server, open the /etc/wireguard/wg0.conf file with vim or your preferred editor again.

```console
$ vim /etc/wireguard/wg0.conf
```
At the bottom of the file after the `SaveConfig = true` line, paste the following lines:

```txt
PostUp = ufw route allow in on wg0 out on eth0
PostUp = iptables -t nat -I POSTROUTING -o eth0 -j MASQUERADE
PostUp = ip6tables -t nat -I POSTROUTING -o eth0 -j MASQUERADE
PreDown = ufw route delete allow in on wg0 out on eth0
PreDown = iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE
PreDown = ip6tables -t nat -D POSTROUTING -o eth0 -j MASQUERADE
```

The PostUp lines will run when the WireGuard Server starts the virtual VPN tunnel. In the example here, it will add three ufw and iptables rules:
- `ufw route allow in on wg0 out on eth0` - This rule will allow forwarding IPv4 and IPv6 traffic that comes in on the wg0 VPN interface to the eth0 network interface on the server. It works in conjunction with the net.ipv4.ip_forward and net.ipv6.conf.all.forwarding sysctl values that you configured in the previous section.
- `iptables -t nat -I POSTROUTING -o eth0 -j MASQUERADE` - This rule configures masquerading, and rewrites IPv4 traffic that comes in on the wg0 VPN interface to make it appear like it originates directly from the WireGuard Server’s public IPv4 address.
- `ip6tables -t nat -I POSTROUTING -o eth0 -j MASQUERADE` - This rule configures masquerading, and rewrites IPv6 traffic that comes in on the wg0 VPN interface to make it appear like it originates directly from the WireGuard Server’s public IPv6 address.

> Note: You can run these command just after running the server without adding Post and Pre configs.

The PreDown rules run when the WireGuard Server stops the virtual VPN tunnel. These rules are the inverse of the PostUp rules, and function to undo the forwarding and masquerading rules for the VPN interface when the VPN is stopped.

In both cases, edit the configuration to include or exclude the IPv4 and IPv6 rules that are appropriate for your VPN. For example, if you are just using IPv4, then you can exclude the lines with the ip6tables commands.

Conversely, if you are only using IPv6, then edit the configuration to only include the ip6tables commands. The ufw lines should exist for any combination of IPv4 and IPv6 networks. Save and close the file when you are finished.

The last part of configuring the firewall on your WireGuard Server is to allow traffic to and from the WireGuard UDP port itself. If you did not change the port in the server’s `/etc/wireguard/wg0.conf` file, the port that you will open is 51820. If you chose a different port when editing the configuration be sure to substitute it in the following UFW command.

> Note: In my experience, data centers might close irregular ports such as the default WireGuard port `51820`, so I suggest choosing a popular service port for your VPN connection instead. I usually prefer database or streaming-service ports, since heavy data traffic on those ports looks less suspicious. (i.e. MongoDB's default port, 27017)

In case you forgot to open the SSH port when following the prerequisite tutorial, add it here too:

```console
$ ufw allow 51820/udp # The chosen VPN server port (you can change it to what ever you want)
$ ufw allow OpenSSH   # To be able to connect the server using openSSH trough port 22
```

> Note: If you are using a different firewall or have customized your UFW configuration, you may need to add additional firewall rules. For example, if you decide to tunnel all of your network traffic over the VPN connection, you will need to ensure that port 53 traffic is allowed for DNS requests, and ports like 80 and 443 for HTTP and HTTPS traffic respectively. If there are other protocols that you are using over the VPN then you will need to add rules for them as well.

After adding those rules, disable and re-enable UFW to restart it and load the changes from all of the files you’ve modified:
```console
$ ufw disable
$ ufw enable
```

You can confirm the rules are in place by running the ufw status command. Run it, and you should receive output like the following:

```console
$ ufw status
```
```output
Status: active

To                         Action      From
--                         ------      ----
51280/udp                  ALLOW       Anywhere                  
22/tcp                     ALLOW       Anywhere                  
51280/udp (v6)             ALLOW       Anywhere (v6)             
22/tcp (v6)                ALLOW       Anywhere (v6)
```

Your WireGuard Server is now configured to correctly handle the VPN’s traffic, including forwarding and masquerading for peers. With the firewall rules in place, you can start the WireGuard service itself to listen for peer connections.

## 3.3 Starting the WireGuard Server
WireGuard can be configured to run as a systemd service using its built-in wg-quick script. While you could manually use the wg command to create the tunnel every time you want to use the VPN, doing so is a manual process that becomes repetitive and error prone. Instead, you can use systemctl to manage the tunnel with the help of the wg-quick script.

Using a systemd service means that you can configure WireGuard to start up at boot so that you can connect to your VPN at any time as long as the server is running. To do this, enable the wg-quick service for the wg0 tunnel that you’ve defined by adding it to systemctl:
```terminal
$ systemctl enable wg-quick@wg0
```
Now start the service:

```console
$ systemctl start wg-quick@wg0
```
Double check that the WireGuard service is active with the following command. You should see active (running) in the output:
```console
$ systemctl status wg-quick@wg0.service
```
```output
● wg-quick@wg0.service - WireGuard via wg-quick(8) for wg0
     Loaded: loaded (/lib/systemd/system/wg-quick@.service; enabled; vendor preset: enabled)
     Active: active (exited) since Wed 2021-08-25 15:24:14 UTC; 5s ago
       Docs: man:wg-quick(8)
             man:wg(8)
             https://www.wireguard.com/
             https://www.wireguard.com/quickstart/
             https://git.zx2c4.com/wireguard-tools/about/src/man/wg-quick.8
             https://git.zx2c4.com/wireguard-tools/about/src/man/wg.8
    Process: 3245 ExecStart=/usr/bin/wg-quick up wg0 (code=exited, status=0/SUCCESS)
   Main PID: 3245 (code=exited, status=0/SUCCESS)

Aug 25 15:24:14 wg0 wg-quick[3245]: [#] wg setconf wg0 /dev/fd/63
Aug 25 15:24:14 wg0 wg-quick[3245]: [#] ip -4 address add 10.8.0.1/24 dev wg0
Aug 25 15:24:14 wg0 wg-quick[3245]: [#] ip -6 address add fd0d:86fa:c3bc::1/64 dev wg0
Aug 25 15:24:14 wg0 wg-quick[3245]: [#] ip link set mtu 1420 up dev wg0
Aug 25 15:24:14 wg0 wg-quick[3245]: [#] ufw route allow in on wg0 out on eth0
Aug 25 15:24:14 wg0 wg-quick[3279]: Rule added
Aug 25 15:24:14 wg0 wg-quick[3279]: Rule added (v6)
Aug 25 15:24:14 wg0 wg-quick[3245]: [#] iptables -t nat -I POSTROUTING -o eth0 -j MASQUERADE
Aug 25 15:24:14 wg0 wg-quick[3245]: [#] ip6tables -t nat -I POSTROUTING -o eth0 -j MASQUERADE
Aug 25 15:24:14 wg0 systemd[1]: Finished WireGuard via wg-quick(8) for wg0.
```

The output shows the ip commands that are used to create the virtual wg0 device and assign it the IPv4 and IPv6 addresses that you added to the configuration file. You can use these rules to troubleshoot the tunnel, or with the wg command itself if you would like to try manually configuring the VPN interface.

With the server configured and running, the next step is to configure your client machine as a WireGuard Peer and connect to the WireGuard Server. WireGuard clients are available for almost every popular operating system, such as Windows, Linux, Android, iOS, macOS, and many more. You can simply download the proper client, load the configuration file you downloaded from wireguard-ui, and connect to the server. ([**Download the WireGuard client**](https://www.wireguard.com/install/))

> Note: when ever you want to add more clients to the server, just run wireguard-ui and add your clients. After terminating the wireguard-ui, you have to restart the wireguard service using `systemctl restart wg-quick@wg0`. If you didn't add Post and Pre Scripts to the wireguard config file like the previous section, you have to run iptables MASQUERADE rules again.

# 5. Reverse Proxy
Congratulations, by now you've configured a virtual private network for yourself. But you might still not be able to connect to it directly, since the government restricts users from connecting to the outside world, and your VPN server sits squarely outside it. To let your clients escape the local intranet, you need a second machine, one inside a local data center that's connected to the public internet. That machine becomes your middle server, the bridge to the VPN server you configured previously. The simplest way to use it as a bridge is to set up a proxy on that middle server which redirects requests to the target VPN server. In computer networking, a proxy server is a server application that acts as an intermediary between a client requesting a resource and the server providing that resource ([Wikipedia](https://en.wikipedia.org/wiki/Proxy_server)).

## 5.1 Nginx Reverse Proxy
Nginx is a popular web-server application that is used to deploy various web applications and it has so many capabilities. One of the configurations that you can set for Nginx is to redirect incoming requests to a specific address by setting proxy routes. As we know that WireGuard traffic is a stream of data and its UDP. So, we have to set a stream proxy route for our purpose.
<div align=center>
    <img width="60%" src="/assets/images/posts/blog/vpn-setup/nginx-logo.png"/>
</div>
<br>

This time, let's connect to our middle server using ssh and after updating its packages, install Nginx on that machine.

```console
$ apt install nginx --yes
```

After that, open Nginx configuration file from `/etc/nginx/nginx.conf` using vim or your preferred editor:

```console
$ vim /etc/nginx/nginx.conf
```

Then add a stream section at the end of that file and write your proxy config there:

```
stream {
    server {
        listen 51820 udp;
        proxy_pass 77.222.67.140:51820;
    }
}
```

Save and exit when you are done. Restart nginx:

```console
$ service nginx restart
```

> Note: In the config above, I opened 51820 port on my middle server and redirected incoming requests through this port to the target VPN server which we have wireguard installed. The first port number on the config `51820` is the port that I want to open on my middle server and the other one is the port I chose for my VPN server. You have to change these numbers if you chose something else.

<div align=center>
    <img src="/assets/images/posts/blog/vpn-setup/wg-client.png"/>
</div>
<br>

Now your clients should be able to connect to the VPN server through the middle server by changing the `Endpoint` on their configuration.

```txt
[Interface]
PrivateKey = [CLIENT_PRIVATE_KEY]
Address = 10.252.1.1/32
DNS = 1.1.1.1

[Peer]
PublicKey = [PUBLIC_KEY]
PresharedKey = [PRE_SHARED_KEY]
AllowedIPs = 0.0.0.0/0
Endpoint = 77.222.67.140:51820 --> change this address to  YOUR_MIDDLE_SERVER_IP:51820
PersistentKeepalive = 15
```

After editing the config file on the client's machines:

```txt
[Interface]
PrivateKey = [CLIENT_PRIVATE_KEY]
Address = 10.252.1.1/32
DNS = 1.1.1.1

[Peer]
PublicKey = [PUBLIC_KEY]
PresharedKey = [PRE_SHARED_KEY]
AllowedIPs = 0.0.0.0/0
Endpoint = 192.168.0.1:51820
PersistentKeepalive = 15
```

## 5.2 Nginx on docker
Since an Nginx Docker image is available on Docker Hub, you can use the Nginx container instead of installing Nginx on a separate VM. Running your proxy server this way is cheaper than renting a whole virtual machine for it. There are also proxy managers on Docker Hub with a web-based interface, such as the popular [**nginx-proxy-manager**](https://nginxproxymanager.com/guide/#project-goal).
<div align=center>
    <img width="50%" src="/assets/images/posts/blog/vpn-setup/docker-logo.png"/>
</div>
<br>

# 6. WireGuard Over TCP (Optional)
WireGuard itself only works over UDP, since it aims to be as fast as possible, but some providers may block UDP packets with their firewalls. This causes problems and prevents WireGuard from working properly. To tackle this issue, we have to convert UDP packets into TCP packets and transfer them through the network after delivering the packets to the VPN server, revert the TCP packets into UDP and pass them to the WireGuard service. This goal can be reached using [**udptunnel**](http://www1.cs.columbia.edu/~lennox/udptunnel/). ([**udp2raw**]() is another popular UDP to TCP converter)

> Note: Udptunnel does not support IPv6.

Udptunnel is a simple application written in C, and it needs to run on both endpoints. On one end, it listens for incoming UDP packets on a specific port and converts them to TCP packets, then transfers those TCP packets to the destination address, where a udptunnel server on the other endpoint is running. In our case, the first endpoint is our middle server, the local server inside the local intranet. The second endpoint, obviously, is our machine out in the open world.

<div align=center>
    <img src="/assets/images/posts/blog/vpn-setup/udp2raw.svg"/>
</div>
<br>


Udptunnel can be run in two modes server mode and client mode. You have to run the first instance in client mode and the second one as the server. In server mode it listens for TCP packets and transfers them back into UDP and in client mode receives UDP packets and transfers them to TCP.

Let's start with the server, the machine located in the open world. First, we have to download the source code of udptunnel:

```console
$ wget https://github.com/rfc1036/udptunnel/archive/refs/heads/master.zip
```

Then we have to build the source code using the build-essential **make** but before that, run the command below to be sure you have the required packages:

```console
$ sudo apt install build-essential pkg-config zip unzip -y
```

Let's unzip the downloaded source codes:
```console
$ unzip master.zip
```

And build the source files:

```console
$ cd udptunnle-master
$ sudo make
$ sudo make install 
```

Now let's pick a TCP port and open the port on the firewall (I chose port 8080):

```console
$ sudo ufw allow 8080/tcp
$ sudo ufw disable
$ sudo ufw enable
```

Then run the udptunnel as server:
```console
$ udptunnel --server 0.0.0.0:8080 --verbose 127.0.0.1:51820
```
It listens to the incoming TCP packets from everywhere converts the into UDP and passes them through port 51820 on the localhost which is the WireGuard server port.

Now jump into the middle server and, download and build the udptunnel source codes:

```console
$ sudo apt install build-essential pkg-config zip unzip --yes
$ wget https://github.com/rfc1036/udptunnel/archive/refs/heads/master.zip
$ unzip master.zip
$ cd udptunnle-master
$ sudo make
$ sudo make install 
```

Pick a udp port, (I chose 8080 again!)

```console
$ sudo ufw allow 8080
$ sudo ufw disable
$ sudo ufw enable
```

And run the udptunnel client on the middle server:

```
$ udptunnel  0.0.0.0:8080 77.222.67.140:8080
```

> Note: In the code above, `77.222.67.140` was the outer machine's address.

That's it, go change the endpoint address to the new UDP port on the middle server:

```txt
[Interface]
PrivateKey = [CLIENT_PRIVATE_KEY]
Address = 10.252.1.1/32
DNS = 1.1.1.1

[Peer]
PublicKey = [PUBLIC_KEY]
PresharedKey = [PRE_SHARED_KEY]
AllowedIPs = 0.0.0.0/0
Endpoint = 192.168.0.1:51820 --> change this address to  YOUR_MIDDLE_SERVER_IP:8080
PersistentKeepalive = 15
```

After editing the config file on the client's machines:

```txt
[Interface]
PrivateKey = [CLIENT_PRIVATE_KEY]
Address = 10.252.1.1/32
DNS = 1.1.1.1

[Peer]
PublicKey = [PUBLIC_KEY]
PresharedKey = [PRE_SHARED_KEY]
AllowedIPs = 0.0.0.0/0
Endpoint = 192.168.0.1:8080
PersistentKeepalive = 15
```

That's it.

> Note: udptunnel is not a background server which means you don't have to exit the ssh sessions (on middle and outer servers) while running this application. You can build a systemctl daemon, use [screen](https://linuxhint.com/linux-screen-command-tutorial/) or [tmux](https://en.wikipedia.org/wiki/Tmux) for running this application to prevent crashing of udptunnel after closing the session.

And remember using WireGuard over TCP will affect your connection speed/bandwidth but some networks block or reduce the bandwidth for UDP connections and we have no other choices but running wire guard over TCP. Here is my experience using WireGuard over TCP:

<p align=center>
    <img src="/assets/images/posts/blog/vpn-setup/speed-test-udp-no-filter.png"/>
    <br>
    <span>Bandwidth while using UDP and it's not restricted by government's firewall</span>
</p>

<p align=center>
    <img src="/assets/images/posts/blog/vpn-setup/speed-test-udp.png"/>
    <br>
    <span>Bandwidth while using UDP and it's restricted by government's firewall</span>

</p>

<p align=center>
    <img src="/assets/images/posts/blog/vpn-setup/speed-test-tcp.png"/>
    <br>
    <span>Bandwidth while using WireGuard over TCP</span>

</p>

# 7. Final words
Some of you might run into trouble with this solution because your government filters specific ports. If things aren't working, try changing the chosen port numbers and try again. In some cases, government firewalls might block UDP packets entirely, in which case you'll need the solution from [part 6](#6-wireguard-over-tcp-(optional)).

The reverse proxy could also be done with some simple firewall (iptables) rules, but I chose Nginx so as not to get bogged down in iptables' complexities.

WireGuard is, as of writing, the newest and best VPN protocol out there: secure, fast, and easy to configure, which is exactly why I chose it for this article. But in extreme cases it can still be detected, and your servers could get blocked.

To reduce that risk, keep the number of your clients as low as possible, to limit the network traffic flowing through these servers. You can also apply this same abstract solution to other VPN protocols and find the one that works best for you (such as [**Shadowsocks**](https://shadowsocks.org/), [**Google Outline**](https://getoutline.org/), [**OpenVPN**](https://openvpn.net/), [**Softether**](https://www.softether.org/), [**V2Ray**](https://www.v2ray.com/), etc.). If you run into trouble with this article, reading through the references below and doing a bit of searching on DuckDuckGo (or Google) should help you find a solution. I hope you found this article useful.

for a better world,<br>
Regards

# References
- [How to set up wireguard on ubuntu 20.04 (LTS)](https://www.digitalocean.com/community/tutorials/how-to-set-up-wireguard-on-ubuntu-20-04)
- [Wireguard-ui](https://github.com/ngoduykhanh/wireguard-ui)
- [Wireguard vpn and nginx reverse proxy](https://ericiniguez.com/p/wireguard-vpn-and-nginx-reverse-proxy/)
- [Nginx proxy manager](https://nginxproxymanager.com/guide/#project-goal)
- [WireGuard + udptunnel](https://www.oilandfish.com/posts/wireguard-udptunnel.html)