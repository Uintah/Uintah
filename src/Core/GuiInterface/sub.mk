# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/GuiInterface

SRCS     += $(SRCDIR)/DebugSettings.cc $(SRCDIR)/GuiManager.cc \
	$(SRCDIR)/GuiServer.cc $(SRCDIR)/Histogram.cc \
	$(SRCDIR)/MemStats.cc $(SRCDIR)/Remote.cc $(SRCDIR)/TCL.cc \
	$(SRCDIR)/TCLInit.cc $(SRCDIR)/TCLTask.cc $(SRCDIR)/GuiVar.cc \
	$(SRCDIR)/ThreadStats.cc \
	$(SRCDIR)/TCLstrbuff.cc \

PSELIBS := Core/Exceptions Core/Util Core/Thread \
	Core/Containers Core/TkExtensions
LIBS := $(TCL_LIBRARY) $(ITK_LIBRARY) $(X11_LIBS)

include $(SRCTOP)/scripts/smallso_epilogue.mk

