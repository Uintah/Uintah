#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/Datatypes/CS684

SRCS     += $(SRCDIR)/DRaytracer.cc $(SRCDIR)/ImageR.cc \
	$(SRCDIR)/Pixel.cc $(SRCDIR)/RTPrims.cc $(SRCDIR)/RadPrims.cc \
	$(SRCDIR)/Sample2D.cc $(SRCDIR)/Scene.cc $(SRCDIR)/Spectrum.cc

PSELIBS := SCICore/Persistent SCICore/Geometry SCICore/Math \
	SCICore/Thread SCICore/Exceptions SCICore/Geom \
	SCICore/Containers SCICore/Datatypes
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:35:55  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:19  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
