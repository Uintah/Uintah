# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Domain

SRCS     += \
	$(SRCDIR)/Extractor.cc\
        $(SRCDIR)/Register.cc\
#	$(SRCDIR)/DomainManager.cc\

PSELIBS := Dataflow/Network Core/Datatypes Dataflow/Widgets \
	Core/Persistent Core/Exceptions Core/Thread \
	Core/Containers Core/TclInterface Core/Geom \
	Core/Datatypes Core/Geometry Core/TkExtensions \
	Core/Util 

LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

