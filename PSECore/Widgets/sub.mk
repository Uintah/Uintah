#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := PSECore/Widgets

SRCS     += $(SRCDIR)/ArrowWidget.cc $(SRCDIR)/BaseWidget.cc \
	$(SRCDIR)/BoxWidget.cc $(SRCDIR)/CriticalPointWidget.cc \
	$(SRCDIR)/CrosshairWidget.cc $(SRCDIR)/FrameWidget.cc \
	$(SRCDIR)/GaugeWidget.cc $(SRCDIR)/LightWidget.cc \
	$(SRCDIR)/PathWidget.cc $(SRCDIR)/PointWidget.cc \
	$(SRCDIR)/RingWidget.cc $(SRCDIR)/ScaledBoxWidget.cc \
	$(SRCDIR)/ScaledFrameWidget.cc $(SRCDIR)/ViewWidget.cc

PSELIBS := PSECore/Datatypes PSECore/Constraints SCICore/Exceptions \
	SCICore/Geom SCICore/Thread SCICore/TclInterface \
	SCICore/Containers SCICore/Geometry
LIBS := -lm

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:28:04  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
