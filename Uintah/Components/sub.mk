#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Uintah/Components


SUBDIRS :=  $(SRCDIR)/ICE $(SRCDIR)/MPM \
	$(SRCDIR)/Arches $(SRCDIR)/ProblemSpecification \
	$(SRCDIR)/Schedulers $(SRCDIR)/SimulationController

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.5  2000/04/19 21:20:00  dav
# more MPI stuff
#
# Revision 1.4  2000/04/11 07:10:34  sparker
# Completing initialization and problem setup
# Finishing Exception modifications
#
# Revision 1.3  2000/04/07 23:02:14  sparker
# Fixed arches compile
#
# Revision 1.2  2000/03/20 19:38:17  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:24  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
