#
# Makefile fragment for Packages/Morgan/Core/PackagePathIterator
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Morgan/Core/PackagePathIterator

CFLAGS += -DDEFAULT_PACKAGE_PATH=\"$(PACKAGE_PATH)\"
SRCS     += $(SRCDIR)/PackagePathIterator.cc

PSELIBS := 
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk
