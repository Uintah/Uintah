# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/MIT/Dataflow/Modules/Metropolis

SRCS     += \
	$(SRCDIR)/PriorPart.cc\
	$(SRCDIR)/LikelihoodPart.cc\
        $(SRCDIR)/SamplerInterface.cc\
        $(SRCDIR)/Sampler.cc\
        $(SRCDIR)/SamplerGui.cc\
        $(SRCDIR)/Bayer.cc\
#[INSERT NEW CODE FILE HERE]


PSELIBS := Packages/MIT/Core/Datatypes \
	Dataflow/Network Dataflow/Ports \
	Core/2d \
	Core/Datatypes \
	Core/Parts \
	Core/PartsGui \
	Core/Algorithms/DataIO \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions 
        Core/Geom Core/Datatypes Core/Geometry \
LIBS := -L/usr/local/lib \
	$(LAPACK_LIBRARY)  $(BLAS_LIBRARY) -lcvode -lunuran $(FLIBS) -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


