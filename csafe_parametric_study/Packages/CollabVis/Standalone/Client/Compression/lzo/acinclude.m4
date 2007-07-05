## --------------------------------------------------------- ##
## My own customization.                                     ##
## --------------------------------------------------------- ##

# serial 2

AC_DEFUN(mfx_CUSTOMIZE,
[
mfx_PROG_CC_BUG_SIGNED_TO_UNSIGNED_CASTING
mfx_PROG_GCC_BUG_SCHEDULE_INSNS
mfx_PROG_GCC_BUG_STRENGTH_REDUCE

dnl /***********************************************************************
dnl // Prepare some macros
dnl ************************************************************************/

CFLAGS_GCC_OMIT_FRAME_POINTER=""
if test "$GCC" = yes; then
  CFLAGS_GCC_OMIT_FRAME_POINTER="-fomit-frame-pointer"
  if test "$mfx_cv_prog_checkergcc" = yes; then
    CFLAGS_GCC_OMIT_FRAME_POINTER="-fno-omit-frame-pointer"
  fi
  if test "$enable_debug" = yes; then
    CFLAGS_GCC_OMIT_FRAME_POINTER="-fno-omit-frame-pointer"
  fi
  if test "$enable_profiling" = yes; then
    CFLAGS_GCC_OMIT_FRAME_POINTER="-fno-omit-frame-pointer"
  fi
  if test "$enable_coverage" = yes; then
    CFLAGS_GCC_OMIT_FRAME_POINTER="-fno-omit-frame-pointer"
  fi
fi
AC_SUBST(CFLAGS_GCC_OMIT_FRAME_POINTER)dnl

if test "$enable_debug" = yes; then
  test "$ac_cv_prog_cc_g" = yes && CFLAGS="$CFLAGS -g"
  test "$ac_cv_prog_cxx_g" = yes && CXXFLAGS="$CXXFLAGS -g"
fi


dnl /***********************************************************************
dnl // Compiler and architecture specific stuff
dnl ************************************************************************/

AC_SUBST(M_CC)dnl
AC_SUBST(M_ARCH)dnl
AC_SUBST(M_CPU)dnl

M_CC="unknown"
M_ARCH="unknown"
M_CPU="$host_cpu"

if test "$GCC" = yes; then
  M_CC="GCC"
  if test "$enable_debug" = yes; then
    CFLAGS_O="-O0"
  else
    CFLAGS_O="-O2"
  fi
  CFLAGS_W="-Wall -Wcast-align -Wwrite-strings"
  case $host in
    i[[34567]]86-*)
      M_ARCH="i386"
      mfx_unaligned_ok_2="yes"
      mfx_unaligned_ok_4="yes"
      CFLAGS_O="$CFLAGS_O -fno-strength-reduce"
      ;;
    *)
      if test "$mfx_cv_prog_gcc_bug_strength_reduce" = yes; then
        CFLAGS_O="$CFLAGS_O -fno-strength-reduce"
      fi
      ;;
  esac
  if test "$mfx_cv_prog_gcc_bug_schedule_insns" = yes; then
    CFLAGS_O="$CFLAGS_O -fno-schedule-insns -fno-schedule-insns2"
  fi
else
  CFLAGS_O="$CFLAGS_O"
# CFLAGS_O="$CFLAGS_O -O"
  CFLAGS_W="$CFLAGS_W"
fi

AC_DEFINE_UNQUOTED(MFX_ARCH,"$M_ARCH")
AC_DEFINE_UNQUOTED(MFX_CPU,"$host_cpu")

M_ARCH=`echo "$M_ARCH" | sed -e 's/[^a-zA-Z0-9]//g'`
M_CPU=`echo "$M_CPU" | sed -e 's/[^a-zA-Z0-9]//g'`
])

