#!/bin/bash
#______________________________________________________________________
#
#  git-push-notify.sh
#
#  Purpose:
#    Poll a GitHub repository and send one email per new commit to a
#    mailing list, formatted like the old svn commit emails:
#
#      Author / Date / New Revision header, Added/Deleted/Modified
#      file lists, the log message, then a unified diff per file.
#
#    Git has no revision numbers, so short commit hashes are used where
#    svn printed "rev N".  A Branch line and a link to the commit on
#    GitHub are added to the header.  Branch deletions get a short
#    notice; force-pushes are flagged on the Branch line.
#
#    The script keeps a bare clone in MIRROR_DIR.  Each run records the
#    branch tips, fetches, and compares.  The first run only creates the
#    clone and sends nothing.
#
#  Usage:
#    git-push-notify.sh            (from cron, e.g. every 5 minutes)
#
#      */5 * * * * /home/rt/scripts/git-push-notify.sh
#
#  Dependencies:
#    git, flock (util-linux), awk, sendmail (an MTA on this host that
#    can deliver mail, e.g. postfix relaying through smtp.utah.edu)
#______________________________________________________________________

#__________________________________
# Configuration

REPO_URL="https://github.com/Uintah/uintah.git"
WEB_URL="https://github.com/Uintah/uintah"
REPO_NAME="Uintah/uintah"
STATE_DIR="/var/lib/git-push-notify"
MIRROR_DIR="${STATE_DIR}/uintah.git"
LOCK_FILE="${STATE_DIR}/uintah.lock"
MAIL_TO="uintah-commits@lists.utah.edu"
MAIL_SENDER="rt@uintah-build.chpc.utah.edu"     # envelope sender; bounces go here
MAIL_FROM="${MAIL_SENDER}"
SENDMAIL="/usr/sbin/sendmail"
MAX_COMMITS=50          # per branch per run; the rest get one summary email
MAX_DIFF_LINES=500     # per email; longer diffs are truncated

EMPTY_TREE="4b825dc642cb6eb9a060e54bf8d69288fbee4904"
SEPARATOR="==================================================================="

#__________________________________
# Print an error to stderr and exit

die()
{
  local msg="$1"
  echo "  ERROR: ${msg}" >&2
  exit 1
}

#__________________________________
# Run git against the mirror

mgit()
{
  git -C "${MIRROR_DIR}" -c core.quotePath=false "$@"
}

#__________________________________
# Send a message (subject, body file, optional Reply-To) to the list

sendMessage()
{
  local subject="$1"
  local bodyFile="$2"
  local replyTo="$3"
  local rc

  {
    echo "From: ${MAIL_FROM}"
    echo "To: ${MAIL_TO}"
    echo "Subject: ${subject}"
    if [[ -n "${replyTo}" ]]; then
      echo "Reply-To: ${replyTo}"
    fi
    echo "MIME-Version: 1.0"
    echo "Content-Type: text/plain; charset=UTF-8"
    echo ""
    cat "${bodyFile}"
  } | "${SENDMAIL}" -t -f "${MAIL_SENDER}"
  rc=$?

  if [[ $rc -ne 0 ]]; then
    echo "  ERROR: sendmail failed (rc=${rc}) for: ${subject}" >&2
    return $rc
  fi
  return 0
}

#__________________________________
# Print one file's diff in svn layout:
#   <Label>: path / URL / ==== / --- path  date (rev old) / +++ ... / hunks

printFileDiff()
{
  local label="$1"
  local path="$2"
  local oldRef="$3"
  local newRef="$4"
  local oldStamp="$5"
  local newStamp="$6"
  local pathHash

  # GitHub anchors each file on a commit page as #diff-<sha256 of path>
  pathHash=$( printf '%s' "${path}" | sha256sum | awk '{ print $1 }' )

  echo "${label}: ${path}"
  echo "URL: ${WEB_URL}/commit/${newRef}#diff-${pathHash}"
  echo "${SEPARATOR}"

  mgit diff --no-color --no-renames -U3 "${oldRef}" "${newRef}" -- "${path}" \
    | awk -v path="${path}"       \
          -v oldStamp="${oldStamp}" \
          -v newStamp="${newStamp}" '
        BEGIN { started = 0 }
        started == 0 && /^Binary files / {
          print "Cannot display: file marked as a binary type."
          exit
        }
        started == 0 && /^@@ / {
          printf "--- %s\t%s\n", path, oldStamp
          printf "+++ %s\t%s\n", path, newStamp
          started = 1
        }
        started == 1 && /^@@ / {
          sub( / @@ .*$/, " @@" )    # drop git function context, as svn does
        }
        started == 1 { print }
      '
  echo ""
}

#__________________________________
# Reply-To for a commit email: the commit author, unless the address
# is a GitHub no-reply alias (replies to those bounce)

commitReplyTo()
{
  local sha="$1"
  local authorEmail

  authorEmail=$( mgit log -1 --format=%ae "${sha}" )

  if [[ -z "${authorEmail}" || "${authorEmail}" == *noreply* ]]; then
    echo ""
  else
    mgit log -1 --format='%an <%ae>' "${sha}"
  fi
}

#__________________________________
# Write the svn-style email body for one commit

writeCommitMail()
{
  local sha="$1"
  local branchLine="$2"
  local bodyFile="$3"
  local parent
  local oldRef
  local oldRev
  local newRev
  local oldDate
  local newDate
  local status
  local path
  local label
  local oldStamp
  local newStamp
  local nameStatus
  local diffFile
  local nLines
  local added=()
  local deleted=()
  local modified=()

  parent=$( mgit rev-parse --verify --quiet "${sha}^1" )

  if [[ -n "${parent}" ]]; then
    oldRef="${parent}"
    oldRev=$( mgit rev-parse --short "${parent}" )
    oldDate=$( TZ=UTC mgit log -1 --format=%cd \
                 --date=format-local:'%Y-%m-%d %H:%M:%S UTC' "${parent}" )
  else
    oldRef="${EMPTY_TREE}"
    oldRev="0"
    oldDate=""
  fi

  newRev=$( mgit rev-parse --short "${sha}" )
  newDate=$( TZ=UTC mgit log -1 --format=%cd \
               --date=format-local:'%Y-%m-%d %H:%M:%S UTC' "${sha}" )

  #__________________________________
  # Changed files, split into svn's three groups

  nameStatus=$( mgit diff-tree -r --no-renames --name-status "${oldRef}" "${sha}" )

  while IFS=$'\t' read -r status path; do
    if [[ -z "${status}" ]]; then
      continue
    fi
    case "${status}" in
      A) added+=( "${path}" ) ;;
      D) deleted+=( "${path}" ) ;;
      *) modified+=( "${path}" ) ;;
    esac
  done <<< "${nameStatus}"

  #__________________________________
  # Header, file lists, log

  {
    mgit log -1 --date=format:'%Y-%m-%d %H:%M:%S %z (%a, %d %b %Y)' \
      --format="Author: %an%nDate: %cd%nNew Revision: %h" "${sha}"
    echo "${branchLine}"
    echo "URL: ${WEB_URL}/commit/${sha}"
    echo ""

    if [[ ${#added[@]} -gt 0 ]]; then
      echo "Added:"
      for path in "${added[@]}"; do
        echo "   ${path}"
      done
    fi
    if [[ ${#deleted[@]} -gt 0 ]]; then
      echo "Deleted:"
      for path in "${deleted[@]}"; do
        echo "   ${path}"
      done
    fi
    if [[ ${#modified[@]} -gt 0 ]]; then
      echo "Modified:"
      for path in "${modified[@]}"; do
        echo "   ${path}"
      done
    fi

    echo "Log:"
    mgit log -1 --format=%B "${sha}" | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}'
    echo ""
  } > "${bodyFile}"

  #__________________________________
  # Per-file diffs (same order as the lists)

  diffFile=$( mktemp )

  for path in "${added[@]}"; do
    oldStamp="(rev 0)"
    newStamp="${newDate} (rev ${newRev})"
    printFileDiff "Added" "${path}" "${oldRef}" "${sha}" "${oldStamp}" "${newStamp}" >> "${diffFile}"
  done

  for path in "${deleted[@]}"; do
    oldStamp="${oldDate} (rev ${oldRev})"
    newStamp="${newDate} (rev ${newRev})"
    printFileDiff "Deleted" "${path}" "${oldRef}" "${sha}" "${oldStamp}" "${newStamp}" >> "${diffFile}"
  done

  for path in "${modified[@]}"; do
    oldStamp="${oldDate} (rev ${oldRev})"
    newStamp="${newDate} (rev ${newRev})"
    printFileDiff "Modified" "${path}" "${oldRef}" "${sha}" "${oldStamp}" "${newStamp}" >> "${diffFile}"
  done

  nLines=$( wc -l < "${diffFile}" )

  if [[ "${nLines}" -gt "${MAX_DIFF_LINES}" ]]; then
    head -n "${MAX_DIFF_LINES}" "${diffFile}" >> "${bodyFile}"
    echo "" >> "${bodyFile}"
    echo "[Diff truncated: ${nLines} lines, first ${MAX_DIFF_LINES} shown.  Full diff:" >> "${bodyFile}"
    echo " ${WEB_URL}/commit/${sha} ]" >> "${bodyFile}"
  else
    cat "${diffFile}" >> "${bodyFile}"
  fi

  rm -f "${diffFile}"
}

#__________________________________
# Mail every new commit on one updated or newly created branch

reportBranch()
{
  local branch="$1"
  local oldSha="$2"
  local newSha="$3"
  shift 3
  local allOldShas=( "$@" )
  local commits
  local count
  local shown=0
  local sha
  local branchLine
  local bodyFile
  local subject
  local logLine
  local rc

  if [[ -z "${oldSha}" ]]; then
    # New branch: commits not reachable from any previously known branch
    commits=$( mgit rev-list --reverse "${newSha}" --not "${allOldShas[@]}" )
    branchLine="Branch: ${branch} (new branch)"
  else
    commits=$( mgit rev-list --reverse "${oldSha}..${newSha}" )
    branchLine="Branch: ${branch}"

    mgit merge-base --is-ancestor "${oldSha}" "${newSha}"
    rc=$?
    if [[ $rc -ne 0 ]]; then
      branchLine="Branch: ${branch} (FORCE-PUSHED, was ${oldSha:0:12})"
    fi
  fi

  count=$( echo "${commits}" | grep -c . )

  while read -r sha; do
    if [[ -z "${sha}" ]]; then
      continue
    fi
    if [[ $shown -ge $MAX_COMMITS ]]; then
      break
    fi

    bodyFile=$( mktemp )
    writeCommitMail "${sha}" "${branchLine}" "${bodyFile}"

    logLine=$( mgit log -1 --format=%s "${sha}" )
    subject="[${REPO_NAME}] ${branch} ${sha:0:9}: ${logLine}"

    sendMessage "${subject}" "${bodyFile}" "$( commitReplyTo "${sha}" )"
    rm -f "${bodyFile}"
    shown=$(( shown + 1 ))
  done <<< "${commits}"

  #__________________________________
  # Too many commits: one summary for the rest

  if [[ $count -gt $shown ]]; then
    bodyFile=$( mktemp )
    {
      echo "Repository: ${REPO_NAME}"
      echo "${branchLine}"
      echo ""
      echo "$(( count - shown )) more commit(s) were pushed but not mailed individually."
      echo "Compare: ${WEB_URL}/compare/${oldSha:0:12}...${newSha:0:12}"
    } > "${bodyFile}"
    sendMessage "[${REPO_NAME}] ${branch}: $(( count - shown )) more commits" "${bodyFile}"
    rm -f "${bodyFile}"
  fi
}

#__________________________________
# Report a deleted branch

reportDeleted()
{
  local branch="$1"
  local oldSha="$2"
  local bodyFile

  bodyFile=$( mktemp )
  {
    echo "Repository: ${REPO_NAME}"
    echo "Branch:     ${branch}"
    echo "Status:     deleted (last commit ${oldSha:0:12})"
  } > "${bodyFile}"

  sendMessage "[${REPO_NAME}] branch ${branch} deleted" "${bodyFile}"
  rm -f "${bodyFile}"
}

#______________________________________________________________________
#______________________________________________________________________

main()
{
  local rc
  local cmd
  local branch
  local sha
  local oldList
  local newList
  local allOldShas=()
  declare -A oldRefs
  declare -A newRefs

  #__________________________________
  # Bulletproofing

  for cmd in git flock awk mktemp sha256sum; do
    command -v "${cmd}" > /dev/null 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      die "required command '${cmd}' not found"
    fi
  done

  if [[ ! -x "${SENDMAIL}" ]]; then
    die "sendmail not found at ${SENDMAIL}"
  fi

  mkdir -p "${STATE_DIR}"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    die "cannot create ${STATE_DIR} (rc=${rc})"
  fi

  #__________________________________
  # Only one copy at a time (a slow fetch must not overlap the next run)

  exec 9> "${LOCK_FILE}"
  flock -n 9
  rc=$?
  if [[ $rc -ne 0 ]]; then
    exit 0
  fi

  #__________________________________
  # First run: create the clone and stop

  if [[ ! -d "${MIRROR_DIR}" ]]; then
    git clone --bare --quiet "${REPO_URL}" "${MIRROR_DIR}"
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "  ERROR: initial clone failed (rc=${rc})" >&2
      exit $rc
    fi
    echo "Initialized ${MIRROR_DIR}; notifications start on the next run."
    exit 0
  fi

  #__________________________________
  # Record branch tips, fetch, record again

  oldList=$( mgit for-each-ref --format='%(refname:short) %(objectname)' refs/heads )

  mgit fetch --prune --quiet origin '+refs/heads/*:refs/heads/*'
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "  ERROR: git fetch failed (rc=${rc})" >&2
    exit $rc
  fi

  newList=$( mgit for-each-ref --format='%(refname:short) %(objectname)' refs/heads )

  while read -r branch sha; do
    if [[ -n "${branch}" ]]; then
      oldRefs["${branch}"]="${sha}"
      allOldShas+=( "${sha}" )
    fi
  done <<< "${oldList}"

  while read -r branch sha; do
    if [[ -n "${branch}" ]]; then
      newRefs["${branch}"]="${sha}"
    fi
  done <<< "${newList}"

  #__________________________________
  # Compare and notify

  for branch in "${!newRefs[@]}"; do
    if [[ "${oldRefs[${branch}]}" != "${newRefs[${branch}]}" ]]; then
      reportBranch "${branch}"             \
                   "${oldRefs[${branch}]}" \
                   "${newRefs[${branch}]}" \
                   "${allOldShas[@]}"
    fi
  done

  for branch in "${!oldRefs[@]}"; do
    if [[ -z "${newRefs[${branch}]}" ]]; then
      reportDeleted "${branch}" "${oldRefs[${branch}]}"
    fi
  done

  exit 0
}

#______________________________________________________________________
#______________________________________________________________________

main "$@"
