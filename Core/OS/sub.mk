#
# Makefile fragment for this subdirectory
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/OS

SRCS     += $(SRCDIR)/Dir.cc $(SRCDIR)/sock.cc

PSELIBS := SCICore/Exceptions
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

