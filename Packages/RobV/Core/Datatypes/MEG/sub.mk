# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/RobV/Core/Datatypes/MEG

SRCS     += \
	$(SRCDIR)/VectorFieldMI.cc

PSELIBS := Dataflow/Network Core/Persistent Core/Geometry \
	Core/Exceptions Core/Datatypes Core/Thread \
	Core/Containers 
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

