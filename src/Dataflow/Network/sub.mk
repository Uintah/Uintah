# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Network

SRCS     += $(SRCDIR)/Connection.cc $(SRCDIR)/ModuleHelper.cc \
	$(SRCDIR)/Network.cc $(SRCDIR)/Port.cc $(SRCDIR)/Module.cc \
	$(SRCDIR)/NetworkEditor.cc $(SRCDIR)/PackageDB.cc \
	$(SRCDIR)/FileUtils.cc $(SRCDIR)/GenFiles.cc\
	$(SRCDIR)/ComponentNode.cc $(SRCDIR)/SkeletonFiles.cc\
        $(SRCDIR)/PackageDBHandler.cc\
	$(SRCDIR)/StrX.cc

PSELIBS := Dataflow/Comm Dataflow/XMLUtil Core/Exceptions Core/Thread \
	Core/Containers Core/GuiInterface Core/Util \
	Core/TkExtensions Core/Geom 
LIBS := $(TCL_LIBRARY) $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

