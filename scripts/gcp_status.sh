#!/usr/bin/env bash
# "Am I being billed for anything right now?" — answered in one command.
#
#   ./scripts/gcp_status.sh          # report
#   ./scripts/gcp_status.sh --off    # report, then delete everything it found
#
# Checks every resource type that bills while idle, not just running VMs.
# The ones that catch people out are the quiet ones: a disk left behind when
# an instance is deleted, a machine image nobody remembers baking, a reserved
# static IP. Those cost money with nothing running at all.
#
# Read-only unless you pass --off.

set -uo pipefail

PROJECT="${PIXIECAD_GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
OFF=0
[ "${1:-}" = "--off" ] && OFF=1

if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  echo "No GCP project set. Run: gcloud config set project <id>" >&2
  exit 2
fi

echo "project: $PROJECT"
echo

FOUND=0

# $1 human label, $2..: gcloud list command
check() {
  local label="$1"; shift
  local out
  out=$("$@" 2>/dev/null)
  if [ -n "$out" ]; then
    FOUND=1
    echo "▶ $label"
    echo "$out" | sed 's/^/    /'
  else
    echo "  none: $label"
  fi
}

check "RUNNING/STOPPED VMs (biggest cost)" \
  gcloud compute instances list "--format=value(name,zone,status,machineType)"
check "Disks (bill even with no VM attached)" \
  gcloud compute disks list "--format=value(name,zone,sizeGb,type,users)"
check "Machine images" \
  gcloud compute machine-images list "--format=value(name)"
check "Snapshots" \
  gcloud compute snapshots list "--format=value(name,diskSizeGb)"
check "Reserved static IPs (billed while unattached)" \
  gcloud compute addresses list "--format=value(name,region,status)"
check "Storage buckets" \
  gcloud storage buckets list "--format=value(name)"

echo
if [ "$FOUND" -eq 0 ]; then
  echo "✅ Nothing billable. You are not being charged for compute or storage."
  exit 0
fi

echo "⚠️  The items above are billing."
if [ "$OFF" -eq 0 ]; then
  echo "   Re-run with --off to delete them, or open:"
  echo "   https://console.cloud.google.com/compute/instances?project=$PROJECT"
  exit 1
fi

# --off: instances first, since deleting one usually takes its disk with it
# and avoids a pointless "disk in use" error afterwards.
echo
echo "deleting everything ..."
gcloud compute instances list --format="value(name,zone)" 2>/dev/null | while read -r n z; do
  [ -n "$n" ] && gcloud compute instances delete "$n" --zone="$z" --quiet
done
gcloud compute disks list --format="value(name,zone)" 2>/dev/null | while read -r n z; do
  [ -n "$n" ] && gcloud compute disks delete "$n" --zone="$z" --quiet
done
gcloud compute machine-images list --format="value(name)" 2>/dev/null | while read -r n; do
  [ -n "$n" ] && gcloud compute machine-images delete "$n" --quiet
done
gcloud compute snapshots list --format="value(name)" 2>/dev/null | while read -r n; do
  [ -n "$n" ] && gcloud compute snapshots delete "$n" --quiet
done
gcloud compute addresses list --format="value(name,region)" 2>/dev/null | while read -r n r; do
  [ -n "$n" ] && gcloud compute addresses delete "$n" --region="$r" --quiet
done

echo
echo "re-checking ..."
exec "$0"
