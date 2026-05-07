#!/usr/bin/env bash
set -euo pipefail

pkg="nbia-data-retriever"
info_dir="/var/lib/dpkg/info"
backup_dir="/tmp/${pkg}-dpkg-script-backup-$(date +%Y%m%d-%H%M%S)"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo:"
  echo "  sudo bash $0"
  exit 1
fi

mkdir -p "${backup_dir}"

for script in prerm postrm postinst; do
  path="${info_dir}/${pkg}.${script}"
  if [[ -e "${path}" ]]; then
    cp -a "${path}" "${backup_dir}/"
    printf '#!/bin/sh\nexit 0\n' > "${path}"
    chmod 755 "${path}"
  fi
done

dpkg --purge "${pkg}"
apt --fix-broken install
dpkg --audit

echo "Backed up original dpkg scripts to: ${backup_dir}"
