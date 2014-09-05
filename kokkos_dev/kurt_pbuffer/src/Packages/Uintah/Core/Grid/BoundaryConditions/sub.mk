# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/Core/Grid/BoundaryConditions

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
	Core/Util \
	Core/Exceptions \
	Core/Geometry \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Util \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/ProblemSpec

LIBS := 
