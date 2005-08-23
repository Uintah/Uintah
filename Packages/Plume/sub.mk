#Makefile fragment for the Packages/Plume directory

include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Packages/Plume

SUBDIRS := \
	$(SRCDIR)/Core \
	$(SRCDIR)/Components \

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 

LIBS := $(BOOST_LIBRARY) $(LOKI_LIBRARY)  $(M_LIBRARY) $(THREAD_LIBRARY)  $(TENA_LIBRARY)
#LIBS := $(OOGL_LIBRARY) $(GL_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY) 

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk

SUBDIRS := $(SRCDIR)/StandAlone
include $(SCIRUN_SCRIPTS)/recurse.mk

#CFLAGS := $(CFLAGS) -OPT:IEEE_arithmetic=3 
#CFLAGS := $(CFLAGS) -OPT:Olimit=16383
#CFLAGS := $(CFLAGS) -OPT:Olimit_opt=on

