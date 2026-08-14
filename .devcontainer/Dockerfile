FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive

# GitHub CLI requires its own APT repository
RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
       -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
       > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    git sudo less procps fzf unzip \
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
