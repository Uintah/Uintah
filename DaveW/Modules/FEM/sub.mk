#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/Modules/FEM

SRCS     += $(SRCDIR)/CStoGeom.cc $(SRCDIR)/CStoSFRG.cc \
	$(SRCDIR)/DipoleMatToGeom.cc $(SRCDIR)/DipoleSourceRHS.cc \
	$(SRCDIR)/ErrorMetric.cc $(SRCDIR)/FieldFromBasis.cc \
	$(SRCDIR)/RecipBasis.cc $(SRCDIR)/RemapVector.cc \
	$(SRCDIR)/VecSplit.cc

PSELIBS := DaveW/Datatypes/General PSECore/Widgets PSECore/Datatypes \
	PSECore/Dataflow SCICore/Persistent SCICore/Exceptions \
	SCICore/Datatypes SCICore/Thread SCICore/TclInterface \
	SCICore/Geom SCICore/Containers SCICore/Geometry 
LIBS := -lm

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:25:45  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
