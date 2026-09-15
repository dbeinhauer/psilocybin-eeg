#!/bin/bash
set -e
iptables -F OUTPUT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 80  -j ACCEPT
iptables -A OUTPUT -j DROP
echo "Firewall initialised."
