# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone/tools/compare_mms


ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/Uintah
else

  PSELIBS := \
        Core/Containers   \
        Core/Exceptions   \
        Core/Geometry     \
        Core/Math         \
        Core/Util         \
        Packages/Uintah/Core/DataArchive \
        Packages/Uintah/Core/Disclosure  \
        Packages/Uintah/Core/Exceptions  \
        Packages/Uintah/Core/Grid        \
        Packages/Uintah/Core/Labels      \
        Packages/Uintah/Core/ProblemSpec                    \
        Packages/Uintah/Core/Util                           \
        Packages/Uintah/CCA/Components/DataArchiver         \
        Packages/Uintah/CCA/Components/Schedulers           \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Packages/Uintah/CCA/Components/PatchCombiner
endif

LIBS := $(BLAS_LIBRARY) $(LAPACK_LIBRARY)

########################################################
# compare_mms

SRCS := $(SRCDIR)/compare_mms.cc \
        $(SRCDIR)/ExpMMS.cc \
        $(SRCDIR)/LinearMMS.cc \
        $(SRCDIR)/SineMMS.cc 

PROGRAM := $(SRCDIR)/compare_mms


include $(SCIRUN_SCRIPTS)/program.mk


########################################################
# compare_scalar

SRCS := $(SRCDIR)/compare_scalar.cc 
PROGRAM := $(SRCDIR)/compare_scalar



include $(SCIRUN_SCRIPTS)/program.mk


compare_mms: prereqs Packages/Uintah/StandAlone/tools/compare_mms/compare_mms
compare_scalar: prereqs Packages/Uintah/StandAlone/tools/compare_mms/compare_scalar
