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
	$(SRCDIR)/PDSimPart.cc\
	$(SRCDIR)/IUniformPDSimPart.cc\
	$(SRCDIR)/RUPDSimPart.cc\
	$(SRCDIR)/IGaussianPDSimPart.cc\
	$(SRCDIR)/ItPDSimPart.cc\
	$(SRCDIR)/RtPDSimPart.cc\
	$(SRCDIR)/MultivariateNormalDSimPart.cc\
	$(SRCDIR)/PriorPart.cc\
	$(SRCDIR)/LikelihoodPart.cc\
        $(SRCDIR)/SamplerInterface.cc\
        $(SRCDIR)/Sampler.cc\
        $(SRCDIR)/SamplerGui.cc\
	$(SRCDIR)/ItPDSimPartGui.cc \
	$(SRCDIR)/RtPDSimPartGui.cc \
	$(SRCDIR)/IGaussianPDSimPartGui.cc \
	$(SRCDIR)/RUPDSimPartGui.cc \
	$(SRCDIR)/IUniformPDSimPartGui.cc \
        $(SRCDIR)/Bayer.cc\
        $(SRCDIR)/AS245.F\
        $(SRCDIR)/mvtpdf.F\
        $(SRCDIR)/mnormpdf.F\
#[INSERT NEW CODE FILE HERE]


PSELIBS := \
	Packages/MIT/Core/Datatypes \
	Dataflow/Network Dataflow/Ports \
	Core/2d \
	Core/Datatypes \
	Core/Parts \
	Core/PartsGui \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions \
        Core/Geom Core/Geometry Core/Thread Core/GuiInterface

LIBS := -L/usr/local/lib \
	$(LAPACK_LIBRARY) \
	 -lcvode -lunuran -lranlib -llinpack \
	$(BLAS_LIBRARY) $(F_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


