#!/bin/bash
#______________________________________________________________________
#  cxx_std.sh
#
#  Purpose:  Report the C++ standard (14, 17, 20, 23 or 26) that a C++
#            compiler uses with a given set of flags, so the Wasatch3P
#            libraries can be built with the same standard as Uintah.
#
#  Usage:    cxx_std.sh <CXX> [CXXFLAGS]
#              <CXX>      C++ compiler, e.g. "mpic++" (may include arguments)
#              CXXFLAGS   compiler flags, e.g. "-O3 -std=c++20"
#
#            On success the standard (e.g. "17") is printed on stdout and
#            the script exits 0.  On failure an error is printed on stderr
#            and the script exits 1.
#
#  Checks:   - the compiler reports __cplusplus (i.e. accepts the flags)
#            - the standard is C++14 or newer (required by Boost.Math)
#            - a -std= request in CXXFLAGS matches what the compiler
#              reports (Intel Classic silently ignores unknown -std values)
#            - cmake is new enough for CMAKE_CXX_STANDARD 23 / 26
#
#  Dependencies: sed, sort -V, cmake
#______________________________________________________________________

#__________________________________
# Print an error to stderr and exit
end_die()
{
  echo "  ERROR(cxx_std.sh): $1" >&2
  exit 1
}

#__________________________________
# Map a __cplusplus value to a standard number
cplusplusToStd()
{
  local cpp=$1

  if [[ $cpp -ge 202400 ]]; then
    echo 26
  elif [[ $cpp -ge 202100 ]]; then
    echo 23
  elif [[ $cpp -ge 201709 ]]; then
    echo 20
  elif [[ $cpp -ge 201703 ]]; then
    echo 17
  elif [[ $cpp -ge 201402 ]]; then
    echo 14
  else
    echo 0
  fi
}

#__________________________________
# Map a -std= value (c++20, gnu++2a, ...) to a standard number
flagToStd()
{
  local name=${1#*++}

  case "$name" in
    14|1y) echo 14 ;;
    17|1z) echo 17 ;;
    20|2a) echo 20 ;;
    23|2b) echo 23 ;;
    26|2c) echo 26 ;;
    *)     echo 0  ;;
  esac
}

#__________________________________
# Return 0 if version $1 >= version $2
versionAtLeast()
{
  local lowest
  lowest=$( printf "%s\n%s\n" "$1" "$2" | sort -V | head -1 )
  [[ "$lowest" == "$2" ]]
}

#______________________________________________________________________
main()
{
  local cxx="$1"
  local cxxFlags="$2"
  local rc

  #__________________________________
  # Bulletproofing
  if [[ -z "$cxx" ]]; then
    end_die "usage: cxx_std.sh <CXX> [CXXFLAGS]"
  fi

  command -v ${cxx%% *} > /dev/null
  rc=$?
  if [[ $rc -ne 0 ]]; then
    end_die "C++ compiler '${cxx%% *}' not found"
  fi

  #__________________________________
  # Ask the compiler which standard it uses with these flags
  local cpp
  cpp=$( echo | $cxx $cxxFlags -dM -E -x c++ - 2> /dev/null | sed -n 's/^#define __cplusplus \([0-9]*\)L$/\1/p' )

  if [[ -z "$cpp" ]]; then
    end_die "'$cxx $cxxFlags' did not report __cplusplus (unsupported -std flag?)"
  fi

  local cxxStd
  cxxStd=$( cplusplusToStd "$cpp" )

  if [[ $cxxStd -eq 0 ]]; then
    end_die "Wasatch3P requires C++14 or newer, but '$cxx $cxxFlags' reports __cplusplus=$cpp"
  fi

  #__________________________________
  # If -std= was requested, it must match what the compiler reports.
  # The last -std= on the line wins, as it does for the compiler.
  local stdFlag
  stdFlag=$( echo " $cxxFlags" | sed -n 's/.* -std=\([^ ]*\).*/\1/p' )

  if [[ -n "$stdFlag" ]]; then
    local requested
    requested=$( flagToStd "$stdFlag" )

    if [[ $requested -ne 0 && $requested -ne $cxxStd ]]; then
      end_die "CXXFLAGS requests -std=$stdFlag but '$cxx' reports C++$cxxStd (__cplusplus=$cpp)"
    fi
  fi

  #__________________________________
  # cmake must understand CMAKE_CXX_STANDARD for 23 and 26
  if [[ $cxxStd -ge 23 ]]; then
    local cmakeVersion
    cmakeVersion=$( cmake --version 2> /dev/null | sed -n 's/^cmake version \([0-9.]*\).*/\1/p' )

    local needed=3.20
    if [[ $cxxStd -ge 26 ]]; then
      needed=3.25
    fi

    if [[ -z "$cmakeVersion" ]]; then
      end_die "cmake not found"
    fi

    versionAtLeast "$cmakeVersion" "$needed"
    rc=$?
    if [[ $rc -ne 0 ]]; then
      end_die "C++$cxxStd requires cmake $needed or newer (found $cmakeVersion)"
    fi
  fi

  echo "$cxxStd"
}

#______________________________________________________________________
#______________________________________________________________________
main "$@"
