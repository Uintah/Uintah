# Makefile fragment for this subdirectory
include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Kurt/Core/Datatypes

SRCS     += \
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Exceptions Core/Geometry \
	Core/Persistent Core/Datatypes \
	Core/Containers  Core/Geom Core/Thread \
	Dataflow/Network Dataflow/XMLUtil \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Datatypes \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Dataflow/Modules/Visualization

LIBS :=  $(LINK) $(XML_LIBRARY) $(GL_LIBS) -lmpi -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

