# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Tester

SRCS     += $(SRCDIR)/PerfTest.cc $(SRCDIR)/RigorousTest.cc

PSELIBS := 
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

