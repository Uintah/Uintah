# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Yarden/Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/Hase.cc\
	$(SRCDIR)/Hase1.cc\
	$(SRCDIR)/IsoSurfaceNOISE.cc\
	$(SRCDIR)/IsoSurfaceSAGE.cc\
	$(SRCDIR)/Isosurface.cc\
	$(SRCDIR)/Noise.cc\
	$(SRCDIR)/Sage.cc\
	$(SRCDIR)/SearchNOISE.cc\
	$(SRCDIR)/Span.cc\
	$(SRCDIR)/ViewTensors.cc\
#	$(SRCDIR)/SageVFem.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Yarden/Datatypes Core/Datatypes Dataflow/Network \
	Core/Persistent Core/Containers Core/Util \
	Core/Exceptions Core/Thread Core/TclInterface \
	Core/Geom Core/Datatypes Core/Geometry \
	Core/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

