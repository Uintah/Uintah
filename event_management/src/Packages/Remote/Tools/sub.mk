# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Packages/Remote/Tools

SUBDIRS := $(SRCDIR)/Util $(SRCDIR)/Math $(SRCDIR)/Model $(SRCDIR)/Image

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := 
LIBS := $(LEX_LIBRARY) $(TIFF_LIBRARY) $(JPEG_LIBRARY) $(M_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

