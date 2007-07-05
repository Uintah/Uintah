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

SRCDIR   := Packages/MIT/Dataflow/Modules/Bayer

SRCS     += \
	$(SRCDIR)/Metropolis.cc\
	$(SRCDIR)/BayerAnalysis.cc\
        $(SRCDIR)/Bayer.cc\
        $(SRCDIR)/bayer.F\
        $(SRCDIR)/bnorm.F\
        $(SRCDIR)/cfode.F\
        $(SRCDIR)/d1mach.F\
        $(SRCDIR)/daxpy.F\
        $(SRCDIR)/ddot.F\
        $(SRCDIR)/dgbfa.F\
        $(SRCDIR)/dgbsl.F\
        $(SRCDIR)/dscal.F\
        $(SRCDIR)/dgefa.F\
        $(SRCDIR)/dgesl.F\
        $(SRCDIR)/ewset.F\
        $(SRCDIR)/fnorm.F\
        $(SRCDIR)/idamax.F\
        $(SRCDIR)/intdy.F\
        $(SRCDIR)/lsoda.F\
        $(SRCDIR)/prja.F\
        $(SRCDIR)/solsy.F\
        $(SRCDIR)/stoda.F\
        $(SRCDIR)/vmnorm.F\
        $(SRCDIR)/xerrwv.F\
        $(SRCDIR)/zufall.F\
#[INSERT NEW CODE FILE HERE]


PSELIBS := Packages/MIT/Core/Datatypes \
	Dataflow/Network Dataflow/Ports \
	Core/2d \
	Core/Datatypes \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/TkExtensions
LIBS := $(GL_LIBRARY) $(TK_LIBRARY) \
        -L/usr/local/lib \
	$(LAPACK_LIBRARY)  $(BLAS_LIBRARY) -lcvode -lunuran $(F_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


