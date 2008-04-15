include $(SCIRUN_SCRIPTS)/so_prologue.mk

SRCDIR  := Packages/Uintah/StandAlone/tools/uda2nrrd

#ifeq ($(findstring teem, $(TEEM_LIBRARY)),teem)
  ifeq ($(LARGESOS),yes)
    PSELIBS := Datflow Packages/Uintah
  else
    PSELIBS := \
        Packages/Uintah/Core/Exceptions    \
        Packages/Uintah/Core/Grid          \
        Packages/Uintah/Core/Util          \
        Packages/Uintah/Core/Math          \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/ProblemSpec   \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/DataArchive   \
        Packages/Uintah/Core/Parallel      \
        Packages/Uintah/CCA/Ports          \
        Packages/Uintah/CCA/Components/ProblemSpecification \
        Core/Basis        \
        Core/Exceptions   \
        Core/Containers   \
        Core/Datatypes    \
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

#endif

include $(SCIRUN_SCRIPTS)/so_epilogue.mk

