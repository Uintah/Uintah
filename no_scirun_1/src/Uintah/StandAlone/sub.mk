# 
# 
# The MIT License
# 
# Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
# Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
# University of Utah.
# 
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
# 
# 
# 
# 
# Makefile fragment for this subdirectory

SRCDIR := Uintah/StandAlone

SUBDIRS := \
        $(SRCDIR)/tools       \
        $(SRCDIR)/Benchmarks

include $(SCIRUN_SCRIPTS)/recurse.mk

##############################################
# sus

# The following variables are used by the Fake* scripts... please
# do not modify...
#
COMPONENTS      = Uintah/CCA/Components
CA              = Uintah/CCA/Components/Arches
ifeq ($(BUILD_ARCHES),yes)
  ARCHES_SUB_LIBS = $(CA)/Mixing $(CA)/fortran $(CA)/Radiation $(CA)/Radiation/fortran
  ifeq ($(BUILD_MPM),yes)
    MPMARCHES_LIB    = $(COMPONENTS)/MPMArches
  endif
  ARCHES_LIBS        = $(COMPONENTS)/Arches
endif
ifeq ($(BUILD_MPM),yes)
  MPM_LIB            = Uintah/CCA/Components/MPM
  ifeq ($(BUILD_ICE),yes)
    MPMICE_LIB       = Uintah/CCA/Components/MPMICE
  endif
endif
ifeq ($(BUILD_ICE),yes)
  ICE_LIB            = Uintah/CCA/Components/ICE
endif


SRCS := $(SRCDIR)/sus.cc

ifeq ($(IS_AIX),yes)
  SET_AIX_LIB := yes
endif

ifeq ($(IS_REDSTORM),yes)
  SET_AIX_LIB := yes
endif

ifeq ($(SET_AIX_LIB),yes)
  AIX_LIBRARY := \
        Core/Containers   \
        Core/Malloc       \
        Core/Math         \
        Core/OS           \
        Core/Persistent   \
        Core/Thread       \
        Core/XMLUtil      \
        Uintah/Core/IO                          \
        Uintah/Core/Math                        \
        Uintah/Core/GeometryPiece               \
        Uintah/CCA/Components/Parent            \
        Uintah/CCA/Components/SwitchingCriteria \
        Uintah/CCA/Components/OnTheFlyAnalysis  \
        Uintah/CCA/Components/Schedulers           \
        Uintah/CCA/Components/SimulationController \
        Uintah/CCA/Components/Solvers              \
        Uintah/CCA/Components/Examples          \
        Uintah/CCA/Components/Angio             \
        $(ARCHES_LIBS)                                   \
        $(MPMARCHES_LIB)                                 \
        $(MPM_LIB)                                       \
        $(ICE_LIB)                                       \
        $(MPMICE_LIB)                                    \
        Uintah/CCA/Components/PatchCombiner     \
        $(ARCHES_SUB_LIBS)
endif

PROGRAM := Uintah/StandAlone/sus

ifeq ($(LARGESOS),yes)
  PSELIBS := Uintah
else
  PSELIBS := \
        Core/Containers   \
        Core/Exceptions   \
        Core/Geometry     \
        Core/Math         \
        Core/Persistent   \
        Core/Thread       \
        Core/Util         \
        Uintah/Core/DataArchive \
        Uintah/Core/Disclosure  \
        Uintah/Core/Exceptions  \
        Uintah/Core/Grid        \
        Uintah/Core/Labels      \
        Uintah/Core/Math        \
        Uintah/Core/Parallel    \
        Uintah/Core/Tracker     \
        Uintah/Core/Util        \
        Uintah/CCA/Ports        \
        Uintah/CCA/Components/Parent \
        Uintah/CCA/Components/Models \
        Uintah/CCA/Components/DataArchiver  \
        Uintah/CCA/Components/LoadBalancers \
        Uintah/CCA/Components/Regridder     \
        Uintah/Core/ProblemSpec             \
        Uintah/CCA/Components/SimulationController \
        Uintah/CCA/Components/Schedulers           \
        Uintah/CCA/Components/ProblemSpecification \
        Uintah/CCA/Components/Solvers              \
        $(AIX_LIBRARY)
endif

ifeq ($(SET_AIX_LIB),yes)
  LIBS := \
        $(TEEM_LIBRARY) \
        $(XML2_LIBRARY) \
        $(Z_LIBRARY) \
        $(THREAD_LIBRARY) \
        $(F_LIBRARY) \
        $(PETSC_LIBRARY) \
        $(HYPRE_LIBRARY) \
        $(BLAS_LIBRARY) \
        $(LAPACK_LIBRARY) \
        $(MPI_LIBRARY) \
        $(X_LIBRARY) \
        $(M_LIBRARY)
else
  LIBS := $(XML2_LIBRARY) $(F_LIBRARY) $(HYPRE_LIBRARY)      \
          $(CANTERA_LIBRARY) $(ZOLTAN_LIBRARY)               \
          $(PETSC_LIBRARY) $(BLAS_LIBRARY) $(LAPACK_LIBRARY) \
          $(MPI_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# parvarRange

SRCS := $(SRCDIR)/partvarRange.cc
PROGRAM := Uintah/StandAlone/partvarRange

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# selectpart

SRCS := $(SRCDIR)/selectpart.cc
PROGRAM := Uintah/StandAlone/selectpart

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Uintah
else
  PSELIBS := \
        Uintah/Core/Exceptions    \
        Uintah/Core/Grid          \
        Uintah/Core/Util          \
        Uintah/Core/Math          \
        Uintah/Core/Parallel      \
        Uintah/Core/Disclosure    \
        Uintah/Core/ProblemSpec   \
        Uintah/Core/Disclosure    \
        Uintah/Core/DataArchive   \
        Uintah/CCA/Ports          \
        Uintah/CCA/Components/ProblemSpecification \
        Core/Containers  \
        Core/Exceptions  \
        Core/Geometry    \
        Core/OS          \
        Core/Persistent  \
        Core/Thread      \
        Core/Util        \
        Core/XMLUtil     
endif

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY) \
	$(TEEM_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# compare_uda

SRCS    := $(SRCDIR)/compare_uda.cc
PROGRAM := Uintah/StandAlone/compare_uda

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Uintah
else
  PSELIBS := \
        Core/XMLUtil  \
        Uintah/Core/Exceptions    \
        Uintah/Core/Grid          \
        Uintah/Core/Util          \
        Uintah/Core/Parallel \
        Uintah/Core/Disclosure    \
        Uintah/Core/Math          \
        Uintah/Core/ProblemSpec   \
        Uintah/Core/DataArchive   \
        Uintah/CCA/Ports          \
        Uintah/CCA/Components/ProblemSpecification \
        Core/Exceptions  \
        Core/Containers  \
        Core/Geometry    \
        Core/OS          \
        Core/Persistent  \
        Core/Thread      \
        Core/Util
endif

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(TEEM_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# slb

SRCS := $(SRCDIR)/slb.cc
PROGRAM := Uintah/StandAlone/slb

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Uintah
else
  PSELIBS := \
	Uintah/Core/Grid \
	Uintah/Core/Util \
	Uintah/Core/GeometryPiece \
	Uintah/Core/Parallel \
	Uintah/Core/Exceptions \
	Uintah/Core/Math \
	Uintah/Core/ProblemSpec \
	Uintah/CCA/Ports \
	Uintah/CCA/Components/ProblemSpecification \
	Core/Exceptions \
        Core/Geometry
endif

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# restart_merger

SRCS := $(SRCDIR)/restart_merger.cc
PROGRAM := Uintah/StandAlone/restart_merger

ifeq ($(LARGESOS),yes)
  PSELIBS := Datflow Uintah
else
  PSELIBS := \
        Uintah/Core/DataArchive   \
        Uintah/Core/Disclosure    \
        Uintah/Core/Exceptions    \
        Uintah/Core/GeometryPiece \
        Uintah/Core/Grid          \
        Uintah/Core/Parallel      \
        Uintah/Core/ProblemSpec   \
        Uintah/Core/Util          \
        Uintah/CCA/Components/DataArchiver         \
        Uintah/CCA/Components/Parent               \
        Uintah/CCA/Components/ProblemSpecification \
        Uintah/CCA/Components/SimulationController \
        Uintah/CCA/Ports          \
        Core/Exceptions  \
        Core/Geometry    \
        Core/Thread      \
        Core/Util        \
        Core/OS          \
        Core/Containers
endif
LIBS    := $(XML2_LIBRARY) $(M_LIBRARY) $(MPI_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# gambitFileReader

SRCS := $(SRCDIR)/gambitFileReader.cc
PROGRAM := Uintah/StandAlone/gambitFileReader

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# Uintah
# Convenience targets for Specific executables 

ifeq ($(BUILD_VISIT),yes)
  VISIT_STUFF=visit_stuff
endif

uintah: sus \
        puda \
        dumpfields \
        compare_uda \
        restart_merger \
        partextract \
        partvarRange \
        selectpart \
        async_mpi_test \
        mpi_test \
        extractV \
        extractF \
        extractS \
        pfs \
        pfs2 \
        gambitFileReader \
        lineextract \
        timeextract \
        faceextract \
        link_inputs \
        link_tools \
        link_regression_tester \
	$(VISIT_STUFF)

###############################################

link_inputs:
	@( if ! test -L Uintah/StandAlone/inputs; then \
               echo "Creating link to inputs directory." ; \
	       ln -sf $(SRCTOP_ABS)/Uintah/StandAlone/inputs Uintah/StandAlone/inputs; \
	   fi )
          
link_orderAccuracy:
	@( if ! test -L Uintah/StandAlone/orderAccuracy; then \
               echo "Creating link to orderAccuracy directory." ; \
	       ln -sf $(SRCTOP_ABS)/Uintah/orderAccuracy Uintah/StandAlone; \
	   fi )          
          
link_tools:
	@( if ! test -L Uintah/StandAlone/puda; then \
               echo "Creating link to all the tools." ; \
	       ln -sf $(OBJTOP_ABS)/Uintah/StandAlone/tools/puda/puda $(OBJTOP_ABS)/Uintah/StandAlone/puda; \
              ln -sf $(OBJTOP_ABS)/Uintah/StandAlone/tools/extractors/lineextract $(OBJTOP_ABS)/Uintah/StandAlone/lineextract; \
              ln -sf $(OBJTOP_ABS)/Uintah/StandAlone/tools/extractors/timeextract $(OBJTOP_ABS)/Uintah/StandAlone/timeextract; \
	   fi )
link_regression_tester:
	@( if ! test -L Uintah/StandAlone/run_RT; then \
               echo "Creating link to regression_tester script." ; \
	       ln -sf $(SRCTOP_ABS)/Uintah/scripts/regression_tester Uintah/StandAlone/run_RT; \
	   fi )

# The REDSTORM portion of the following command somehow prevents Make, on Redstorm,
# from running a bogus compile line of sus...
#
# This is the bogus line:
#
# cc -Minline -O3 -fastsse -fast   -Minform=severe -DREDSTORM  -Llib -lgmalloc \
#       sus.o prereqs Uintah/StandAlone/sus   -o sus
#
# Notice that is using the generic 'cc' compiler, and only a subset of
# the CFLAGS, and the bogus 'prereqs' target that has not been
# expanded...  This happens after _successfully_ running the real link
# line for sus... I have no idea why it is being triggered, but this
# hack seems to prevent the 2nd 'compilation' from running...
#
sus: prereqs Uintah/StandAlone/sus
ifeq ($(IS_REDSTORM),yes)
	@echo "Built sus"
endif

tools: puda dumpfields compare_uda uda2nrrd restart_merger partextract partvarRange selectpart async_mpi_test mpi_test extractV extractF extractS gambitFileReader slb pfs pfs2 timeextract faceextract lineextract compare_mms compare_scalar

puda: prereqs Uintah/StandAlone/tools/puda/puda

dumpfields: prereqs Uintah/StandAlone/tools/dumpfields/dumpfields

compare_uda: prereqs Uintah/StandAlone/compare_uda

restart_merger: prereqs Uintah/StandAlone/restart_merger

partextract: prereqs Uintah/StandAlone/tools/extractors/partextract

partvarRange: prereqs Uintah/StandAlone/partvarRange

selectpart: prereqs Uintah/StandAlone/selectpart

async_mpi_test: prereqs Uintah/StandAlone/tools/mpi_test/async_mpi_test

mpi_test: prereqs Uintah/StandAlone/tools/mpi_test/mpi_test

extractV: prereqs Uintah/StandAlone/tools/extractors/extractV

extractF: prereqs Uintah/StandAlone/tools/extractors/extractF

extractS: prereqs Uintah/StandAlone/tools/extractors/extractS

gambitFileReader: prereqs Uintah/StandAlone/gambitFileReader

slb: prereqs Uintah/StandAlone/slb

pfs: prereqs Uintah/StandAlone/tools/pfs/pfs

pfs2: prereqs Uintah/StandAlone/tools/pfs/pfs2

timeextract: Uintah/StandAlone/tools/extractors/timeextract

faceextract: Uintah/StandAlone/tools/extractors/faceextract

lineextract: Uintah/StandAlone/tools/extractors/lineextract

compare_mms: Uintah/StandAlone/tools/compare_mms/compare_mms

compare_scalar: Uintah/StandAlone/tools/compare_mms/compare_scalar

mpi_test: Uintah/StandAlone/tools/mpi_test/mpi_test
