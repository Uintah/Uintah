#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := Uintah

SUBDIRS := $(SRCDIR)/Components $(SRCDIR)/Datatypes $(SRCDIR)/Exceptions \
	$(SRCDIR)/GUI $(SRCDIR)/Grid $(SRCDIR)/Interface $(SRCDIR)/Math \
	$(SRCDIR)/Modules $(SRCDIR)/Parallel

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := Component PSECore SCICore
LIBS := -lm
ifeq ($(BUILD_PARALLEL),yes)
LIBS := $(LIBS) $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common
endif

include $(SRCTOP)/scripts/largeso_epilogue.mk

SRCS := $(SRCDIR)/sus.cc
PROGRAM := Uintah/sus
ifeq ($(LARGESOS),yes)
PSELIBS := Uintah
else
PSELIBS := Uintah/Grid Uintah/Parallel Uintah/Components/MPM \
	Uintah/Components/DataArchiver \
	Uintah/Components/SimulationController Uintah/Components/ICE \
	Uintah/Components/Schedulers Uintah/Components/Arches \
	Uintah/Components/ProblemSpecification \
	SCICore/Exceptions Uintah/Interface SCICore/Thread
endif
LIBS := $(XML_LIBRARY) -lmpi

include $(SRCTOP)/scripts/program.mk

SRCS := $(SRCDIR)/puda.cc
PROGRAM := Uintah/puda
ifeq ($(LARGESOS),yes)
PSELIBS := Uintah
else
PSELIBS := Uintah/Exceptions Uintah/Grid Uintah/Interface \
	PSECore/XMLUtil SCICore/Exceptions SCICore/Geometry \
	SCICore/Thread SCICore/Util SCICore/OS Uintah/Components/MPM
endif
LIBS 	:= $(XML_LIBRARY)

include $(SRCTOP)/scripts/program.mk

#
# $Log$
# Revision 1.14  2000/07/11 20:00:11  kuzimmer
# Modified so that CCVariables and Matrix3 are recognized
#
# Revision 1.13  2000/06/17 07:06:21  sparker
# Changed ProcessorContext to ProcessorGroup
#
# Revision 1.12  2000/06/15 21:56:55  sparker
# Added multi-patch support (bugzilla #107)
# Changed interface to datawarehouse for particle data
# Particles now move from patch to patch
#
# Revision 1.11  2000/06/08 21:01:42  bigler
# Added SCICore/OS lib to puda build
#
# Revision 1.10  2000/05/30 18:25:50  dav
# added XML_LIBRARY to sus build
#
# Revision 1.9  2000/05/21 08:19:04  sparker
# Implement NCVariable read
# Do not fail if variable type is not known
# Added misc stuff to makefiles to remove warnings
#
# Revision 1.8  2000/05/20 08:09:01  sparker
# Improved TypeDescription
# Finished I/O
# Use new XML utility libraries
#
# Revision 1.7  2000/05/15 19:39:29  sparker
# Implemented initial version of DataArchive (output only so far)
# Other misc. cleanups
#
# Revision 1.6  2000/05/11 20:10:10  dav
# adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
#
# Revision 1.5  2000/05/07 06:01:57  sparker
# Added beginnings of multiple patch support and real dependencies
#  for the scheduler
#
# Revision 1.4  2000/04/11 07:10:29  sparker
# Completing initialization and problem setup
# Finishing Exception modifications
#
# Revision 1.3  2000/03/23 10:30:10  sparker
# Now need to link sus with SCICore/Exceptions
#
# Revision 1.2  2000/03/20 19:38:15  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:22  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
