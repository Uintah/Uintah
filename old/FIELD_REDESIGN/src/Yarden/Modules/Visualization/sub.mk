#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Yarden/Modules/Visualization

SRCS     += \
	$(SRCDIR)/ViewTensors.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Yarden/Datatypes PSECore/Datatypes PSECore/Dataflow \
	SCICore/Persistent SCICore/Containers SCICore/Util \
	SCICore/Exceptions SCICore/Thread SCICore/TclInterface \
	SCICore/Geom SCICore/Datatypes SCICore/Geometry \
	SCICore/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3.2.2  2000/10/26 13:42:53  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.5  2000/10/24 05:58:03  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.4  2000/10/23 23:41:31  yarden
# View Tensors
#
# Revision 1.3  2000/03/20 23:38:40  yarden
# Linux port
#
# Revision 1.2  2000/03/20 19:38:58  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:37  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
