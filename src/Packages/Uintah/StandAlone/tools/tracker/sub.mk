SRCDIR  := Packages/Uintah/StandAlone/tools/tracker
PROGRAM := $(SRCDIR)/tracker

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Packages/Uintah
else
  PSELIBS := \
	Core/Util                     \
        Packages/Uintah/Core/Tracker
endif

SRCS := \
	$(SRCDIR)/TrackerProgram.cc                  


LIBS := $(XML2_LIBRARY) $(TEEM_LIBRARY) $(PNG_LIBRARY) $(Z_LIBRARY) $(M_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

