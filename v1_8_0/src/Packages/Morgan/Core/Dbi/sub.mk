#
# Makefile fragment for Packages/Morgan/Core/Dbi
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Morgan/Core/Dbi

SRCS     += $(SRCDIR)/Dbi.cc $(SRCDIR)/Dbd.cc $(SRCDIR)/ProcDbd.cc

PSELIBS := Core/Exceptions Packages/Morgan/Core/Process
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk
