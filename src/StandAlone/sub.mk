#
#  The MIT License
#
#  Copyright (c) 1997-2017 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
# 
# 
# 
# 
# 
# Makefile fragment for this subdirectory 

SRCDIR := StandAlone

SUBDIRS := \
        $(SRCDIR)/tools       \
        $(SRCDIR)/Benchmarks

include $(SCIRUN_SCRIPTS)/recurse.mk

##############################################
# sus

SRCS := $(SRCDIR)/sus.cc

PROGRAM := StandAlone/sus

ifeq ($(IS_STATIC_BUILD),yes)

  PSELIBS := $(ALL_STATIC_PSE_LIBS)

else

  ifeq ($(LARGESOS),yes)
    PSELIBS := Packages/Uintah
  else
    PSELIBS := $(ALL_PSE_LIBS)
  endif
endif

ifeq ($(IS_STATIC_BUILD),yes)
  LIBS := $(CORE_OS) $(CORE_STATIC_LIBS) $(ZOLTAN_LIBRARY)    \
          $(BOOST_LIBRARY)         \
          $(EXPRLIB_LIBRARY) $(SPATIALOPS_LIBRARY) \
          $(RADPROPS_LIBRARY) $(TABPROPS_LIBRARY) \
          $(PAPI_LIBRARY) $(M_LIBRARY) $(PIDX_LIBRARY) \
          $(POKITT_LIBRARY)
else
  LIBS := $(MPI_LIBRARY) $(XML2_LIBRARY) $(F_LIBRARY) $(HYPRE_LIBRARY)  \
          $(CANTERA_LIBRARY) $(ZOLTAN_LIBRARY)                          \
          $(PETSC_LIBRARY) $(LAPACK_LIBRARY) $(BLAS_LIBRARY)            \
          $(M_LIBRARY) $(THREAD_LIBRARY)                                \
          $(EXPRLIB_LIBRARY) $(SPATIALOPS_LIBRARY)                      \
          $(TABPROPS_LIBRARY) $(RADPROPS_LIBRARY)                       \
          $(BOOST_LIBRARY) $(CUDA_LIBRARY)                              \
          $(PAPI_LIBRARY) $(GPERFTOOLS_LIBRARY) $(PIDX_LIBRARY)
endif

PSELIBS := $(GPU_EXTRA_LINK) $(PSELIBS)

ifeq ($(HAVE_VISIT),yes)
  INCLUDES += $(VISIT_INCLUDE)
  PSELIBS += VisIt/libsim
  LIBS += $(VISIT_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# DigitalFilterGenerator

ifeq ($(BUILD_ARCHES),yes)
  SRCS    := $(SRCDIR)/../CCA/Components/Arches/DigitalFilter/DigitalFilterGenerator.cc
  PROGRAM := StandAlone/DigitalFilterGenerator
  ifneq ($(IS_STATIC_BUILD),yes)
    PSELIBS := $(PSELIBS) Core/IO
  endif

  include $(SCIRUN_SCRIPTS)/program.mk
endif

##############################################
# parvarRange

SRCS := $(SRCDIR)/partvarRange.cc
PROGRAM := StandAlone/partvarRange

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# compare_uda

SRCS    := $(SRCDIR)/compare_uda.cc
PROGRAM := StandAlone/compare_uda

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# slb

SRCS := $(SRCDIR)/slb.cc
PROGRAM := StandAlone/slb

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# restart_merger

#ifeq ($(IS_STATIC_BUILD),yes)
#  PSELIBS := CCA/Components/Parent $(CORE_STATIC_PSELIBS)
#endif

SRCS := $(SRCDIR)/restart_merger.cc
PROGRAM := StandAlone/restart_merger

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# gambitFileReader

SRCS := $(SRCDIR)/gambitFileReader.cc
PROGRAM := StandAlone/gambitFileReader

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# selectpart

SRCS := $(SRCDIR)/selectpart.cc
PROGRAM := StandAlone/selectpart

ifeq ($(IS_STATIC_BUILD),yes)
  PSELIBS := $(CORE_STATIC_PSELIBS)
endif

include $(SCIRUN_SCRIPTS)/program.mk

##############################################
# Uintah
# Convenience targets for Specific executables 

uintah: sus \
        puda \
        dumpfields \
        compare_uda \
        compute_Lnorm_udas \
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
        link_scripts \
        link_tools \
        link_localRT

###############################################

link_inputs:
	@( if ! test -L StandAlone/inputs; then \
               echo "Creating link to inputs directory." ; \
	       ln -sf $(SRCTOP_ABS)/StandAlone/inputs StandAlone/inputs; \
	   fi )

link_scripts:
	@( if ! test -L StandAlone/scripts; then \
               echo "Creating link to scripts directory." ; \
	       ln -sf $(SRCTOP_ABS)/scripts StandAlone/scripts; \
	   fi )

link_orderAccuracy:
	@( if ! test -L StandAlone/orderAccuracy; then \
               echo "Creating link to orderAccuracy directory." ; \
	       ln -sf $(SRCTOP_ABS)/orderAccuracy StandAlone; \
	   fi )          

link_tools:
	@( if ! test -L StandAlone/puda; then \
               echo "Creating link to all the tools." ; \
	       ln -sf $(OBJTOP_ABS)/StandAlone/tools/puda/puda $(OBJTOP_ABS)/StandAlone/puda; \
              ln -sf $(OBJTOP_ABS)/StandAlone/tools/extractors/lineextract $(OBJTOP_ABS)/StandAlone/lineextract; \
              ln -sf $(OBJTOP_ABS)/StandAlone/tools/extractors/timeextract $(OBJTOP_ABS)/StandAlone/timeextract; \
              ln -sf $(OBJTOP_ABS)/StandAlone/tools/extractors/partextract $(OBJTOP_ABS)/StandAlone/partextract; \
	   fi )
link_localRT:
	@( if ! test -L StandAlone/localRT; then \
               echo "Creating link to localRT script." ; \
	       ln -sf $(SRCTOP_ABS)/R_Tester/toplevel/localRT StandAlone/localRT; \
	   fi )

sus: prereqs StandAlone/sus

$(OBJTOP)/StandAlone/sus.o : $(OBJTOP_ABS)/include/svn_info.h

tools: puda dumpfields compare_uda compute_Lnorm_udas restart_merger partextract partvarRange selectpart async_mpi_test mpi_test extractV extractF extractS gambitFileReader slb pfs pfs2 rawToUniqueGrains timeextract faceextract lineextract compare_mms compare_scalar fsspeed

puda: prereqs StandAlone/tools/puda/puda

dumpfields: prereqs StandAlone/tools/dumpfields/dumpfields

compare_uda: prereqs StandAlone/compare_uda

compute_Lnorm_udas: prereqs StandAlone/tools/compute_Lnorm_udas

restart_merger: prereqs StandAlone/restart_merger

partextract: prereqs StandAlone/tools/extractors/partextract

partvarRange: prereqs StandAlone/partvarRange

selectpart: prereqs StandAlone/selectpart

async_mpi_test: prereqs StandAlone/tools/mpi_test/async_mpi_test

mpi_test: prereqs StandAlone/tools/mpi_test/mpi_test

extractV: prereqs StandAlone/tools/extractors/extractV

extractF: prereqs StandAlone/tools/extractors/extractF

extractS: prereqs StandAlone/tools/extractors/extractS

gambitFileReader: prereqs StandAlone/gambitFileReader

slb: prereqs StandAlone/slb

pfs: prereqs StandAlone/tools/pfs/pfs

pfs2: prereqs StandAlone/tools/pfs/pfs2

rawToUniqueGrains: prereqs StandAlone/tools/pfs/rawToUniqueGrains

timeextract: StandAlone/tools/extractors/timeextract

faceextract: StandAlone/tools/extractors/faceextract

lineextract: StandAlone/tools/extractors/lineextract

particle2tiff: StandAlone/tools/extractors/particle2tiff

compare_mms: StandAlone/tools/compare_mms/compare_mms

compare_scalar: StandAlone/tools/compare_mms/compare_scalar

mpi_test: StandAlone/tools/mpi_test/mpi_test

fsspeed: StandAlone/tools/fsspeed/fsspeed
