# Makefile fragment for this subdirectory
include $(SRCTOP)/scripts/smallso_prologue.mk


SRCDIR   := Packages/Kurt/Dataflow/Modules/Vis

SRCS     += \
	$(SRCDIR)/GLTextureBuilder.cc\
	$(SRCDIR)/PadField.cc \
	$(SRCDIR)/TextureVolVis.cc \
	$(SRCDIR)/TexCuttingPlanes.cc \
	$(SRCDIR)/ParticleColorMapKey.cc \
	$(SRCDIR)/RescaleColorMapForParticles.cc \
	$(SRCDIR)/AnimatedStreams.cc \
	$(SRCDIR)/SFRG.cc
#[INSERT NEW CODE FILE HERE]

#		$(SRCDIR)/VolVis.cc \
#		$(SRCDIR)/Packages/KurtScalarFieldReader.cc \
#		$(SRCDIR)/VisControl.cc \
#		$(SRCDIR)/ParticleVis.cc \


PSELIBS :=  Dataflow/Network Core/Datatypes \
        Core/Thread Core/Persistent Core/Exceptions \
        Core/TclInterface Core/Containers Core/Datatypes \
        Core/Geom Core/Geometry Dataflow/Widgets PSECore/XMLUtil \
	Dataflow/Modules/Fields  Kurt/Datatypes SCICore/Util \
	Packages/Uintah/Core/Datatypes Uintah/Grid Uintah/Interface Uintah/Exceptions 


LIBS := $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk


