# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Yarden/Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/IsoSurfaceNOISE.cc\
	$(SRCDIR)/SearchNOISE.cc\
	$(SRCDIR)/Span.cc\
#	$(SRCDIR)/Hase.cc\
#	$(SRCDIR)/Hase1.cc\
#	$(SRCDIR)/IsoSurfaceSAGE.cc\
#	$(SRCDIR)/Isosurface.cc\
#	$(SRCDIR)/Noise.cc\
#	$(SRCDIR)/Sage.cc\
#	$(SRCDIR)/SageVFem.cc\
#	$(SRCDIR)/ViewTensors.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Packages/Yarden/Core/Datatypes Packages/Yarden/Dataflow/Ports \
	Dataflow/Ports Dataflow/Network \
	Core/Persistent Core/Containers Core/Util \
	Core/Exceptions Core/Thread Core/GuiInterface \
	Core/Geom Core/Datatypes Core/Geometry \
	Core/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

