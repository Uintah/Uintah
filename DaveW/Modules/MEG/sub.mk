#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/Modules/MEG

SRCS     += \
	$(SRCDIR)/EleValuesToMatLabFile.cc\
	$(SRCDIR)/FieldCurl.cc\
	$(SRCDIR)/MakeCurrentDensityField.cc\
	$(SRCDIR)/MagneticFieldAtPoints.cc\
	$(SRCDIR)/MagneticScalarField.cc\
	$(SRCDIR)/NegateGradient.cc\
	$(SRCDIR)/SurfToVectGeom.cc\
#[INSERT NEW MODULE HERE]


PSELIBS := DaveW/Datatypes/General PSECore/Dataflow PSECore/Datatypes \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/Datatypes SCICore/TclInterface SCICore/Containers \
	SCICore/Geom
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/06/07 20:54:59  moulding
# made changes to allow the module maker to add to and edit this file
#
# Revision 1.2  2000/03/20 19:36:13  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:51  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
