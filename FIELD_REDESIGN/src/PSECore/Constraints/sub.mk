#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECore/Constraints

SRCS     += $(SRCDIR)/AngleConstraint.cc $(SRCDIR)/BaseConstraint.cc \
	$(SRCDIR)/BaseVariable.cc $(SRCDIR)/CenterConstraint.cc \
	$(SRCDIR)/ConstraintSolver.cc $(SRCDIR)/DistanceConstraint.cc \
	$(SRCDIR)/LineConstraint.cc $(SRCDIR)/PlaneConstraint.cc \
	$(SRCDIR)/ProjectConstraint.cc $(SRCDIR)/PythagorasConstraint.cc \
	$(SRCDIR)/RatioConstraint.cc $(SRCDIR)/SegmentConstraint.cc \
	$(SRCDIR)/VarCore.cc

PSELIBS := SCICore/Containers SCICore/Util SCICore/Exceptions \
	SCICore/Geometry
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:37:16  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:52  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
