# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Packages/Remote/Tools

SUBDIRS := $(SRCDIR)/Util $(SRCDIR)/Math $(SRCDIR)/Model $(SRCDIR)/Image

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := 
LIBS := -lm -L/usr/local/gnu/lib -lfl -ltiff -ljpeg

include $(SRCTOP)/scripts/smallso_epilogue.mk

