# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := Dataflow

SUBDIRS := \
	$(SRCDIR)/Comm \
	$(SRCDIR)/Constraints \
	$(SRCDIR)/Distributed \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Modules \
	$(SRCDIR)/Network \
	$(SRCDIR)/Ports \
	$(SRCDIR)/Widgets \
	$(SRCDIR)/XMLUtil \

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := Core
LIBS := $(TCL_LIBRARY) $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/largeso_epilogue.mk

