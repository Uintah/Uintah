# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := Core/CCA/Component

SUBDIRS := \
	$(SRCDIR)/CIA \
	$(SRCDIR)/PIDL 

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := 
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common

include $(SRCTOP)/scripts/largeso_epilogue.mk

