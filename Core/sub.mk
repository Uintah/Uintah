# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := Core

SUBDIRS := \
	$(SRCDIR)/Containers \
	$(SRCDIR)/Datatypes \
	$(SRCDIR)/Exceptions \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Geom \
	$(SRCDIR)/Geometry \
	$(SRCDIR)/Malloc \
	$(SRCDIR)/Math \
	$(SRCDIR)/OS \
	$(SRCDIR)/Persistent \
	$(SRCDIR)/Process \
	$(SRCDIR)/TclInterface \
	$(SRCDIR)/Tester \
	$(SRCDIR)/Thread \
	$(SRCDIR)/TkExtensions \
	$(SRCDIR)/Util \
#	$(SRCDIR)/Algorithms \

ifeq ($(BUILD_PARALLEL),yes)
SUBDIRS := $(SUBDIRS) $(SRCDIR)/globus_threads
endif

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := 
LIBS := $(BLT_LIBRARY) $(ITCL_LIBRARY) $(TCL_LIBRARY) $(TK_LIBRARY) \
	$(ITK_LIBRARY) $(GL_LIBS) $(GLOBUS_COMMON) $(THREAD_LIBS) -lm -lz

include $(SRCTOP)/scripts/largeso_epilogue.mk

