#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/MPM

SRCS     += $(SRCDIR)/SerialMPM.cc \
	$(SRCDIR)/BoundCond.cc $(SRCDIR)/MPMLabel.cc

SUBDIRS := $(SRCDIR)/ConstitutiveModel $(SRCDIR)/Contact \
	$(SRCDIR)/Fracture \
	$(SRCDIR)/Fracture \
	$(SRCDIR)/ThermalContact \
	$(SRCDIR)/Burn \
	$(SRCDIR)/GeometrySpecification $(SRCDIR)/Util

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := Uintah/Interface Uintah/Grid Uintah/Parallel \
	Uintah/Exceptions SCICore/Exceptions SCICore/Thread \
	SCICore/Geometry PSECore/XMLUtil
LIBS := $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.14  2000/06/20 23:23:02  tan
# Added HeatConduction directory.
#
# Revision 1.13  2000/06/02 23:09:13  jas
# Added Burn directory.
#
# Revision 1.12  2000/05/31 22:20:22  tan
# Added ThermalContact directory.
#
# Revision 1.11  2000/05/30 20:18:59  sparker
# Changed new to scinew to help track down memory leaks
# Changed region to patch
#
# Revision 1.10  2000/05/30 17:55:19  dav
# commited the previous one with a test file.  now it is removed
#
# Revision 1.9  2000/05/30 17:07:56  dav
# I think I added the xerces flag
#
# Revision 1.8  2000/05/26 21:37:30  jas
# Labels are now created and accessed using Singleton class MPMLabel.
#
# Revision 1.7  2000/05/21 08:19:06  sparker
# Implement NCVariable read
# Do not fail if variable type is not known
# Added misc stuff to makefiles to remove warnings
#
# Revision 1.6  2000/05/10 20:02:42  sparker
# Added support for ghost cells on node variables and particle variables
#  (work for 1 patch but not debugged for multiple)
# Do not schedule fracture tasks if fracture not enabled
# Added fracture directory to MPM sub.mk
# Be more uniform about using IntVector
# Made patches have a single uniform index space - still needs work
#
# Revision 1.5  2000/04/26 06:48:12  sparker
# Streamlined namespaces
#
# Revision 1.4  2000/04/12 22:59:03  sparker
# Working to make it compile
# Added xerces to link line
#
# Revision 1.3  2000/03/20 19:38:23  sparker
# Added VPATH support
#
# Revision 1.2  2000/03/20 17:17:05  sparker
# Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
#
# Revision 1.1  2000/03/17 09:29:32  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
