SRCDIR  := Packages/Uintah/StandAlone/tools/uda2nrrd
PROGRAM := $(SRCDIR)/uda2nrrd

#ifeq ($(findstring teem, $(TEEM_LIBRARY)),teem)
  ifeq ($(LARGESOS),yes)
    PSELIBS := Datflow Packages/Uintah
  else
    PSELIBS := \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Packages/Uintah/CCA/Ports          \
        Packages/Uintah/Core/DataArchive   \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/Exceptions    \
        Packages/Uintah/Core/Grid          \
        Packages/Uintah/Core/Math          \
        Packages/Uintah/Core/Parallel      \
        Packages/Uintah/Core/ProblemSpec   \
        Packages/Uintah/Core/Util          \
        Core/Basis        \
        Core/Containers   \
        Core/Datatypes    \
        Core/Exceptions   \
        Core/Geometry     \
        Core/Math         \
        Core/Persistent   \
        Core/Thread       \
        Core/Util         \
        Core/XMLUtil
  endif

  SRCS := \
	$(SRCDIR)/bc.cc                  \
	$(SRCDIR)/build.cc               \
	$(SRCDIR)/handleVariable.cc      \
	$(SRCDIR)/uda2nrrd.cc            \
	$(SRCDIR)/update_mesh_handle.cc  \
	$(SRCDIR)/particles.cc           \
	$(SRCDIR)/wrap_nrrd.cc 

  LIBS := $(XML2_LIBRARY) $(TEEM_LIBRARY) $(PNG_LIBRARY) $(Z_LIBRARY) $(M_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY)

  include $(SCIRUN_SCRIPTS)/program.mk

#endif

uda2nrrd: prereqs Packages/Uintah/StandAlone/tools/uda2nrrd/uda2nrrd

