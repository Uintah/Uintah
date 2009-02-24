# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Uintah/CCA/Components/Arches/MCRT/ArchesRMCRT

SRCS     += $(SRCDIR)/RMCRTRadiationModel.cc \
            $(SRCDIR)/RMCRTnoInterpolation.cc \
            $(SRCDIR)/Surface.cc \
            $(SRCDIR)/RealSurface.cc \
            $(SRCDIR)/TopRealSurface.cc \
            $(SRCDIR)/BottomRealSurface.cc \
    	      $(SRCDIR)/FrontRealSurface.cc \
            $(SRCDIR)/BackRealSurface.cc \
            $(SRCDIR)/LeftRealSurface.cc \
            $(SRCDIR)/RightRealSurface.cc \
	          $(SRCDIR)/VirtualSurface.cc \
            $(SRCDIR)/ray.cc \
            $(SRCDIR)/VolElement.cc \
            $(SRCDIR)/MakeTableFunction.cc
             

# SUBDIRS := $(SRCDIR)/fortran 
# include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
        Uintah/Core/ProblemSpec   \
        Uintah/Core/GeometryPiece \
        Uintah/Core/Grid          \
        Uintah/Core/Util          \
        Uintah/Core/Disclosure    \
        Uintah/Core/Exceptions    \
        Uintah/CCA/Components/OnTheFlyAnalysis \
        Uintah/CCA/Ports     \
        Uintah/Core/Parallel \
        Core/Util       \
        Core/Thread     \
        Core/Exceptions \
        Core/Geometry   \
        Core/Containers

LIBS := $(LIBS) $(XML2_LIBRARY) $(F_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

