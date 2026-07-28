#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:-data/canonical/keyword_v1/annotation/v1}"
ARCHIVE_FILE="${2:-new_plan/artifacts/annotation_blind_packet_v1.tar.gz}"
LOG_FILE="${3:-new_plan/logs/p0_annotation_export.log}"

PRIMARY_FILE="${SOURCE_DIR}/annotation_primary.csv"
SECONDARY_FILE="${SOURCE_DIR}/annotation_secondary.csv"
MANIFEST_FILE="${SOURCE_DIR}/annotation_manifest.json"
TEACHER_FILE="${SOURCE_DIR}/review_only_teacher_reference.jsonl"

EXPECTED_PRIMARY_SHA256="cc7db664c1075323f91dd81d3d7bb942871075a9c98f1c6282dd0b5459e1536e"
EXPECTED_SECONDARY_SHA256="b9966dec3bc3bfc60a09ae53bcb65746faec8baef3370f2efac9f4ff8d64796b"
EXPECTED_TEACHER_SHA256="e7c19eaa339e3e534ad2c0b6db9dc4c6b3c35f317083a11371dba8a1c755d2fb"

mkdir -p "$(dirname "${ARCHIVE_FILE}")" "$(dirname "${LOG_FILE}")"
: > "${LOG_FILE}"

log() {
  printf '%s\n' "$1" | tee -a "${LOG_FILE}"
}

log "{\"event\":\"annotation_export_start\",\"source_dir\":\"${SOURCE_DIR}\",\"archive_file\":\"${ARCHIVE_FILE}\"}"

for required_file in "${PRIMARY_FILE}" "${SECONDARY_FILE}" "${MANIFEST_FILE}" "${TEACHER_FILE}"; do
  if [ ! -f "${required_file}" ]; then
    log "{\"event\":\"annotation_export_missing_file\",\"path\":\"${required_file}\",\"status\":\"FAIL\"}"
    exit 1
  fi
done

primary_sha256="$(sha256sum "${PRIMARY_FILE}" | awk '{print $1}')"
secondary_sha256="$(sha256sum "${SECONDARY_FILE}" | awk '{print $1}')"
teacher_sha256="$(sha256sum "${TEACHER_FILE}" | awk '{print $1}')"

log "{\"event\":\"annotation_export_hashes\",\"primary_sha256\":\"${primary_sha256}\",\"secondary_sha256\":\"${secondary_sha256}\",\"teacher_sha256\":\"${teacher_sha256}\"}"

if [ "${primary_sha256}" != "${EXPECTED_PRIMARY_SHA256}" ] || \
   [ "${secondary_sha256}" != "${EXPECTED_SECONDARY_SHA256}" ] || \
   [ "${teacher_sha256}" != "${EXPECTED_TEACHER_SHA256}" ]; then
  log '{"event":"annotation_export_hash_check","status":"FAIL","reason":"frozen_hash_mismatch"}'
  exit 1
fi

# Only blind files are exported. The teacher reference is checked above but
# deliberately excluded from the archive to prevent accidental label leakage.
tar -C "${SOURCE_DIR}" -czf "${ARCHIVE_FILE}" \
  annotation_primary.csv \
  annotation_secondary.csv \
  annotation_manifest.json

archive_sha256="$(sha256sum "${ARCHIVE_FILE}" | awk '{print $1}')"
archive_bytes="$(wc -c < "${ARCHIVE_FILE}" | tr -d ' ')"

log "{\"event\":\"annotation_export_complete\",\"status\":\"PASS\",\"archive_file\":\"${ARCHIVE_FILE}\",\"archive_sha256\":\"${archive_sha256}\",\"archive_bytes\":${archive_bytes},\"teacher_reference_in_archive\":false}"
