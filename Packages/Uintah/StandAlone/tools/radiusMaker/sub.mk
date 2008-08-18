SRCDIR  := Packages/Uintah/StandAlone/tools/radiusMaker
PROGRAM := $(SRCDIR)/radius_maker

#ifeq ($(findstring teem, $(TEEM_LIBRARY)),teem)
  ifeq ($(LARGESOS),yes)
    PSELIBS := Datflow Packages/Uintah
  else
    PSELIBS := \
        Core/Math         \
        Core/XMLUtil
  endif

  SRCS := \
	$(SRCDIR)/radius_maker.cc

  LIBS := $(TEEM_LIBRARY) $(BLAS_LIBRARY) $(LAPACK_LIBRARY)

  include $(SCIRUN_SCRIPTS)/program.mk

#endif

radius_maker: prereqs Packages/Uintah/StandAlone/tools/radiusMaker/radius_maker

