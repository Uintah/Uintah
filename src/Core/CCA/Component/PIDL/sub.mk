#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Component/PIDL

SRCS     += $(SRCDIR)/GlobusError.cc $(SRCDIR)/InvalidReference.cc \
	$(SRCDIR)/MalformedURL.cc $(SRCDIR)/Object.cc \
	$(SRCDIR)/Object_proxy.cc $(SRCDIR)/PIDL.cc \
	$(SRCDIR)/ProxyBase.cc $(SRCDIR)/Reference.cc \
	$(SRCDIR)/ReplyEP.cc $(SRCDIR)/ServerContext.cc \
	$(SRCDIR)/TypeInfo.cc $(SRCDIR)/TypeInfo_internal.cc \
	$(SRCDIR)/URL.cc $(SRCDIR)/Wharehouse.cc

PSELIBS := SCICore/Exceptions SCICore/Thread
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:25:13  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
