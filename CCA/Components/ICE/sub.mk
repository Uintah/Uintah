# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/ICE
#---------------------
# remove rate form files
#
RF= $(RateForm)
ifeq ($(RF),true)
SRCS     += \
	$(SRCDIR)/ICERF.cc \
       $(SRCDIR)/ICEDebugRF.cc
       
 else
SRCS     += \
	$(SRCDIR)/ICE.cc \
       $(SRCDIR)/ICEDebug.cc
endif
#---------------------

SRCS	+= \
	$(SRCDIR)/ICELabel.cc    \
	$(SRCDIR)/ICEMaterial.cc \
	$(SRCDIR)/GeometryObject2.cc \
 	$(SRCDIR)/MathToolbox.cc \

SUBDIRS := $(SRCDIR)/EOS $(SRCDIR)/Advection

include $(SCIRUN_SCRIPTS)/recurse.mk          

PSELIBS := \
	Packages/Uintah/CCA/Components/HETransformation \
	Packages/Uintah/CCA/Ports                       \
	Packages/Uintah/Core/Grid                       \
	Packages/Uintah/Core/Math                       \
	Packages/Uintah/Core/Disclosure                 \
	Packages/Uintah/Core/ProblemSpec                \
	Packages/Uintah/Core/Parallel                   \
	Packages/Uintah/Core/Exceptions                 \
	Packages/Uintah/Core/Math                       \
	Core/Exceptions Core/Thread                     \
	Core/Geometry Dataflow/XMLUtil                  \
	Core/Datatypes

LIBS	:= $(XML_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk



