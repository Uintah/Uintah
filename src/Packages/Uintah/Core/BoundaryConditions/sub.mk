# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/BoundaryConditions

SRCS     += \
	$(SRCDIR)/VelocityBoundCond.cc \
	$(SRCDIR)/TemperatureBoundCond.cc \
	$(SRCDIR)/PressureBoundCond.cc \
	$(SRCDIR)/DensityBoundCond.cc \
	$(SRCDIR)/MassFracBoundCond.cc \
	$(SRCDIR)/BoundCondFactory.cc \
	$(SRCDIR)/BoundCondReader.cc \
	$(SRCDIR)/BCData.cc \
	$(SRCDIR)/BCDataArray.cc \
	$(SRCDIR)/BCGeomBase.cc \
	$(SRCDIR)/UnionBCData.cc \
	$(SRCDIR)/DifferenceBCData.cc \
	$(SRCDIR)/SideBCData.cc \
	$(SRCDIR)/CircleBCData.cc \
	$(SRCDIR)/AnnulusBCData.cc \
	$(SRCDIR)/RectangleBCData.cc

PSELIBS := \

LIBS := 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

