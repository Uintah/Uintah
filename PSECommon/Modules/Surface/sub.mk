#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Surface

SRCS     += $(SRCDIR)/GenSurface.cc $(SRCDIR)/LabelSurface.cc \
	$(SRCDIR)/SFUGtoSurf.cc $(SRCDIR)/SurfGen.cc \
	$(SRCDIR)/SurfInterpVals.cc $(SRCDIR)/SurfNewVals.cc \
	$(SRCDIR)/SurfToGeom.cc

PSELIBS := PSECore/Dataflow PSECore/Datatypes PSECore/Widgets \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/Containers SCICore/TclInterface SCICore/Geometry \
	SCICore/Datatypes SCICore/Geom
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:37:05  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:23  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
