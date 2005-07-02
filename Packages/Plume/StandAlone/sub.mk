# Makefile fragment for this subdirectory

# Plume
SRCDIR := Packages/Plume/StandAlone

SRCS := $(SRCDIR)/Plume.cc \
	Config.cc

ifeq ($(LARGESOS),yes)
  PSELIBS := Core/CCA Packages/Plume
else
  PSELIBS := Core/Exceptions Core/CCA/Comm \
        Core/CCA/PIDL Core/CCA/spec \
	SCIRun Core/CCA/SSIDL Core/Thread
endif

PROGRAM := Packages/Plume/StandAlone/Plume
include $(SCIRUN_SCRIPTS)/program.mk

# Convenience target:
#.PHONY: Plume
#Plume: prereqs Packages/Plume/StandAlone/Plume 
#.PHONY: libPlume
#libPlume: lib/libPackages_Plume_Core.so
