#
# Makefile fragment for this subdirectory
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECore/Dataflow

SRCS     += $(SRCDIR)/Connection.cc $(SRCDIR)/ModuleHelper.cc \
	$(SRCDIR)/Network.cc $(SRCDIR)/Port.cc $(SRCDIR)/Module.cc \
	$(SRCDIR)/NetworkEditor.cc $(SRCDIR)/PackageDB.cc \
	$(SRCDIR)/FileUtils.cc $(SRCDIR)/GenFiles.cc\
	$(SRCDIR)/ComponentNode.cc $(SRCDIR)/SkeletonFiles.cc\
        $(SRCDIR)/PackageDBHandler.cc\
	$(SRCDIR)/StrX.cc

PSELIBS := PSECore/Comm SCICore/Exceptions SCICore/Thread \
	SCICore/Containers SCICore/TclInterface SCICore/Util \
	SCICore/TkExtensions SCICore/Geom PSECore/XMLUtil
LIBS := $(TCL_LIBRARY) $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

