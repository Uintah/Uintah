#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Domain

SRCS     += \
	$(SRCDIR)/Extractor.cc\
        $(SRCDIR)/Register.cc\
#	$(SRCDIR)/DomainManager.cc\

PSELIBS := PSECore/Dataflow PSECore/Datatypes PSECore/Widgets \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/Containers SCICore/TclInterface SCICore/Geom \
	SCICore/Datatypes SCICore/Geometry SCICore/TkExtensions \
	SCICore/Util 

LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1.2.2  2000/10/26 10:03:25  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.1.2.1  2000/06/07 17:21:54  kuehne
# Added sub.mk
#
# Revision 1.2  2000/03/20 19:36:57  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:01  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
