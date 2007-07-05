include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/rtrt/Core/PathTracer

SRCS += $(SRCDIR)/PathTraceEngine.cc \
	$(SRCDIR)/AmbientOcclusion.cc

PSELIBS :=  \
	Core/Thread Core/Exceptions Core/Persistent Core/Geometry Packages/rtrt/Core

LIBS := $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(GL_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY) $(PNG_LIBRARY) $(Z_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
