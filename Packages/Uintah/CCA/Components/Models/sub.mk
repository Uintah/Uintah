# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Models

SRCS	+= \
       $(SRCDIR)/ModelFactory.cc

SUBDIRS := $(SRCDIR)/test \
           $(SRCDIR)/HEChem
include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
	Core/Exceptions \
	Core/Geometry \
	Core/Thread \
	Core/Util \
        Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Disclosure \
	Packages/Uintah/Core/Exceptions \
        Packages/Uintah/Core/Grid \
        Packages/Uintah/Core/Util \
        Packages/Uintah/Core/GeometryPiece \
        Packages/Uintah/Core/Labels \
	Packages/Uintah/Core/Parallel \
        Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Components/ICE \
	Packages/Uintah/CCA/Components/MPMICE
LIBS	:= 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
