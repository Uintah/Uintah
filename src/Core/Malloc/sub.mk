# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Malloc

SRCS     += $(SRCDIR)/Allocator.cc $(SRCDIR)/AllocOS.cc \
	$(SRCDIR)/malloc.cc $(SRCDIR)/new.cc 

SRCS += $(LOCK_IMPL)

PSELIBS := 
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

