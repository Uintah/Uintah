# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/ICE

SRCS       += $(SRCDIR)/ICE.cc \
       $(SRCDIR)/AMRICE.cc \
       $(SRCDIR)/ICERF.cc \
       $(SRCDIR)/ICEDebug.cc \
       $(SRCDIR)/ICEMaterial.cc \
       $(SRCDIR)/Diffusion.cc \
       $(SRCDIR)/BoundaryCond.cc \
       $(SRCDIR)/Turbulence.cc \
       $(SRCDIR)/SmagorinskyModel.cc \
       $(SRCDIR)/DynamicModel.cc \
       $(SRCDIR)/TurbulenceFactory.cc \
       $(SRCDIR)/impICE.cc \
       $(SRCDIR)/impAMRICE.cc \
       $(SRCDIR)/customInitialize.cc   \
       $(SRCDIR)/ExchangeCoefficients.cc   	
       
SUBDIRS := $(SRCDIR)/EOS $(SRCDIR)/Advection $(SRCDIR)/CustomBCs

include $(SCIRUN_SCRIPTS)/recurse.mk          

PSELIBS := \
       Packages/Uintah/CCA/Components/MPM              \
       Packages/Uintah/CCA/Components/OnTheFlyAnalysis \
       Packages/Uintah/CCA/Ports                       \
       Packages/Uintah/Core/Grid                       \
       Packages/Uintah/Core/Util                       \
       Packages/Uintah/Core/Math                       \
       Packages/Uintah/Core/Disclosure                 \
       Packages/Uintah/Core/GeometryPiece              \
       Packages/Uintah/Core/ProblemSpec                \
       Packages/Uintah/Core/Parallel                   \
       Packages/Uintah/Core/Exceptions                 \
       Packages/Uintah/Core/Math                       \
       Packages/Uintah/Core/Labels                     \
       Core/Exceptions Core/Geometry                   \
       Core/Thread Core/Util

LIBS       := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


