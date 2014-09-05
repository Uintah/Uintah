#
# Makefile fragment for Packages/Morgan/Core/Process
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Morgan/Core/Process

SRCS     += $(SRCDIR)/ProcManager.cc $(SRCDIR)/Proc.cc 

PSELIBS := Core/Exceptions Packages/Morgan/Core/PackagePathIterator
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk
