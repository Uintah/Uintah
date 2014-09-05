#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Component/PIDL

SRCS     += $(SRCDIR)/GlobusError.cc $(SRCDIR)/InvalidReference.cc \
	$(SRCDIR)/MalformedURL.cc $(SRCDIR)/Object.cc \
	$(SRCDIR)/Object_proxy.cc $(SRCDIR)/PIDL.cc \
	$(SRCDIR)/PIDLException.cc \
	$(SRCDIR)/ProxyBase.cc $(SRCDIR)/Reference.cc \
	$(SRCDIR)/ReplyEP.cc $(SRCDIR)/ServerContext.cc \
	$(SRCDIR)/TypeInfo.cc $(SRCDIR)/TypeInfo_internal.cc \
	$(SRCDIR)/URL.cc $(SRCDIR)/Wharehouse.cc

PSELIBS := SCICore/Exceptions SCICore/Thread
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/03/23 20:43:07  sparker
# Added copy ctor to all exception classes (for Linux/g++)
#
# Revision 1.2  2000/03/20 19:35:47  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:13  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
