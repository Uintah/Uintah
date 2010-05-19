# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := CCA/Components/Arches/MCRT/ArchesRMCRT

SRCS     += $(SRCDIR)/RMCRTRadiationModel.cc \
            $(SRCDIR)/RMCRTRRSDStratified.cc \
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
            $(SRCDIR)/MakeTableFunction.cc \
						$(SRCDIR)/Consts.cc


# SUBDIRS := $(SRCDIR)/fortran 
# include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
        Core/ProblemSpec   \
        Core/GeometryPiece \
        Core/Grid          \
        Core/Util          \
        Core/Disclosure    \
        Core/Exceptions    \
        CCA/Components/OnTheFlyAnalysis \
        CCA/Ports     \
        Core/Parallel \
        Core/Util       \
        Core/Thread     \
        Core/Exceptions \
        Core/Geometry   \
        Core/Containers \
	\
	Core/Math

LIBS := $(LIBS) $(XML2_LIBRARY) $(F_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

