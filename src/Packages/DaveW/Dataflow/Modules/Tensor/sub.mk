#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/Modules/Tensor

SRCS     += $(SRCDIR)/Bundles.cc $(SRCDIR)/Flood.cc \
	$(SRCDIR)/TensorAccessFields.cc $(SRCDIR)/TensorAnisotropy.cc

PSELIBS := DaveW/Datatypes/General PSECore/Datatypes PSECore/Dataflow \
	PSECore/Widgets SCICore/Persistent SCICore/Geometry \
	SCICore/Math SCICore/Exceptions SCICore/Datatypes \
	SCICore/Thread SCICore/Geom SCICore/Containers \
	SCICore/TclInterface 
LIBS := -lm

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:26:04  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
