#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
mkdir -p deps
cd deps
apt-get download libeigen3-dev libmpfr-dev libgmp-dev
for package in ./*.deb; do
    dpkg-deb -x "$package" .
done
