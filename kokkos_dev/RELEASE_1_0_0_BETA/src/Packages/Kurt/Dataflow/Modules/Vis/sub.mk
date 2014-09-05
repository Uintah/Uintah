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
#		$(SRCDIR)/KurtScalarFieldReader.cc \
#		$(SRCDIR)/VisControl.cc \
#		$(SRCDIR)/ParticleVis.cc \


PSELIBS := \
	Dataflow/Network \
        Core/Thread Core/Persistent Core/Exceptions \
        Core/GuiInterface Core/Containers Core/Datatypes \
        Core/Geom Core/Geometry Dataflow/Widgets \
	Datatypes/XMLUtil \
	Dataflow/Modules/Fields Core/Util \
	Packages/Kurt/Datatypes           \
	Packages/Uintah/Core/Datatypes    \
	Packages/Uintah/Core/Grid         \
	Packages/Uintah/CCA/Ports         \
	Packages/Uintah/Core/Exceptions 


LIBS := $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk


