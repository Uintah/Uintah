#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Uintah

SUBDIRS := $(SRCDIR)/Components $(SRCDIR)/Datatypes $(SRCDIR)/Exceptions \
	$(SRCDIR)/GUI $(SRCDIR)/Grid $(SRCDIR)/Interface $(SRCDIR)/Math \
	$(SRCDIR)/Modules $(SRCDIR)/Parallel

include $(OBJTOP_ABS)/scripts/recurse.mk

PSELIBS := Component PSECore SCICore
LIBS := -lm
ifeq ($(BUILD_PARALLEL),yes)
LIBS := $(LIBS) $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common
endif

include $(OBJTOP_ABS)/scripts/largeso_epilogue.mk

SRCS := $(SRCDIR)/sus.cc
PROGRAM := Uintah/sus
ifeq ($(LARGESOS),yes)
PSELIBS := Uintah
LIBS :=
else
PSELIBS := Uintah/Parallel Uintah/Components/MPM \
	Uintah/Components/SimulationController Uintah/Components/ICE \
	Uintah/Components/Schedulers Uintah/Components/Arches
LIBS :=
endif

include $(OBJTOP_ABS)/scripts/program.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:29:22  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
