#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/Modules/FEM

SRCS     += \
	$(SRCDIR)/CStoGeom.cc\
	$(SRCDIR)/CStoSFRG.cc\
	$(SRCDIR)/DipoleMatToGeom.cc\
	$(SRCDIR)/DipoleSourceRHS.cc\
	$(SRCDIR)/ErrorMetric.cc\
	$(SRCDIR)/FieldFromBasis.cc\
	$(SRCDIR)/RecipBasis.cc\
	$(SRCDIR)/RemapVector.cc\
	$(SRCDIR)/VecSplit.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := DaveW/Datatypes/General PSECore/Widgets PSECore/Datatypes \
	PSECore/Dataflow SCICore/Persistent SCICore/Exceptions \
	SCICore/Datatypes SCICore/Thread SCICore/TclInterface \
	SCICore/Geom SCICore/Containers SCICore/Geometry 
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/06/07 20:54:58  moulding
# made changes to allow the module maker to add to and edit this file
#
# Revision 1.2  2000/03/20 19:36:10  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:45  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
