# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/CCA/Component/PIDL

SRCS     += \
	$(SRCDIR)/GlobusError.cc \
	$(SRCDIR)/InvalidReference.cc \
	$(SRCDIR)/MalformedURL.cc \
	$(SRCDIR)/Object.cc \
	$(SRCDIR)/Object_proxy.cc \
	$(SRCDIR)/PIDL.cc \
	$(SRCDIR)/PIDLException.cc \
	$(SRCDIR)/ProxyBase.cc \
	$(SRCDIR)/Reference.cc \
	$(SRCDIR)/ReplyEP.cc \
	$(SRCDIR)/ServerContext.cc \
	$(SRCDIR)/TypeInfo.cc \
	$(SRCDIR)/TypeInfo_internal.cc \
	$(SRCDIR)/URL.cc \
	$(SRCDIR)/Warehouse.cc

PSELIBS := Core/Exceptions Core/Thread
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common

include $(SRCTOP)/scripts/smallso_epilogue.mk

