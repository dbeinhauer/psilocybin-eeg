FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl sudo less procps fzf zsh unzip gnupg2 \
    gh iptables ipset iproute2 \
    build-essential libgl1 libglib2.0-0 \
    libxcb-xinerama0 libxkbcommon-x11-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Node.js for Claude Code
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Non-root user (required for --dangerously-skip-permissions)
RUN useradd -m -s /bin/bash researcher \
    && echo "researcher ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" \
       > /etc/sudoers.d/researcher-firewall \
    && chmod 0440 /etc/sudoers.d/researcher-firewall

RUN npm install -g @anthropic-ai/claude-code

RUN mkdir -p /home/researcher/.claude && chown researcher:researcher /home/researcher/.claude

WORKDIR /workspace

COPY .devcontainer/init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/init-firewall.sh

USER researcher

ENV SHELL=/bin/bash
ENV MPLBACKEND=Agg
ENV PSILOCYBIN_DATA_DIR=/data
ENV PSILOCYBIN_RESULTS_DIR=/results
