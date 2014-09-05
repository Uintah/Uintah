#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


# Epilogue fragment for subdirectories.  This is included from
# either smallso_epilogue.mk or largeso_epilogue.mk

OBJS := $(patsubst %.c,%.o,$(filter %.c,$(SRCS))) \
	   $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS))) \
	   $(patsubst %.s,%.o,$(filter %.s,$(SRCS))) \
	   $(patsubst %.F,%.o,$(filter %.F,$(SRCS))) \
	   $(patsubst %.f,%.o,$(filter %.f,$(SRCS))) \
	   $(patsubst %.fif,%.o,$(filter %.fif,$(SRCS))) \
	   $(patsubst %.y,%.o,$(filter %.y,$(SRCS)))

LIBNAME := $(LIBDIR)/lib$(subst /,_,$(SRCDIR)).$(SO_OR_A_FILE)

# We always link against the internal Dataflow malloc
ifneq ($(SRCDIR),Core/Malloc)
ifeq ($(LARGESOS),yes)
else
PSELIBS := $(PSELIBS) $(MALLOCLIB)
endif
endif

# The libraries are specified like Core/Thread but get
# name-mangled to Core_Thread
PSELIBS := $(subst /,_,$(PSELIBS))
ifneq ($(REPOSITORY_FLAGS),)
REPOSITORIES_$(LIBNAME) := $(REPOSITORY_FLAGS) $(SRCDIR)/ptrepository $(patsubst %,$(REPOSITORY_FLAGS) %/ptrepository, $(PSELIBS))
endif

#  These targets will be used to "make all"
ALLTARGETS := $(ALLTARGETS) $(LIBNAME)
ALLSRCS := $(ALLSRCS) $(SRCS)

# At recurse.mk time all sub.mk files are included, so we need to 
# tuck the value of $(LIBS) away in a mangled variable name
$(notdir $(LIBNAME)_LIBS) := $(LIBS)

# The dependencies can be either .o files or .so files.  The .so
# files are other shared libraries within the PSE.  This allows
# the dependencies between .so's to be expressed conveiently.
# There are two complicated substitions here.  The first creates the
# library depdencies by transforming something like Core_Thread
# to lib/libCore_Thread.so.  The second transforms it from
# lib/libCore_Thread.so to -lCore_Thread.  This is so that
# we can use the -l syntax to link, but still express the dependicies.
ifeq ($(NEED_SONAME),yes)
  SONAMEFLAG = -Wl,-soname -Wl,$(notdir $@)
else
  SONAMEFLAG = 
endif

TMPPSELIBS = $(patsubst %,lib%.$(SO_OR_A_FILE),$(PSELIBS)) 
TMPP = $(patsubst libPackages_%,PACKAGE%,$(TMPPSELIBS))
TMP = $(patsubst lib%,SCIRUN%,$(TMPP))
TMP_CORE_PSELIBS = $(patsubst SCIRUN%,lib%,$(TMP))
CORE_PSELIBS = $(patsubst PACKAGE%,,$(TMP_CORE_PSELIBS))
TMP_PACK_PSELIBS = $(patsubst PACKAGE%,libPackages_%,$(TMP))
PACK_PSELIBS = $(patsubst SCIRUN%,,$(TMP_PACK_PSELIBS))

$(LIBNAME): $(OBJS) $(patsubst %,$(SCIRUN_LIBDIR)/%,$(CORE_PSELIBS)) $(patsubst %,$(LIBDIR)/%,$(PACK_PSELIBS))
	rm -f $@
  ifeq ($(CC),newmpxlc)
	ar -v -q $@ $(filter %.o,$^)
  else
  ifeq ($(IS_OSX),yes)
	$(CXX) $(SCI_THIRDPARTY_LIBRARY) $(LDFLAGS) $(SOFLAGS) -install_name $(SCIRUN_LIBDIR_ABS)/$(patsubst lib/%,%,$@) -o $@ $(SONAMEFLAG) $(filter %.o,$^) $(patsubst $(SCIRUN_LIBDIR)/lib%.so,-l%,$(filter %.$(SO_OR_A_FILE),$^)) $(REPOSITORIES_$@) $($(notdir $@)_LIBS) $(TAU_MPI_LIBS) $(TAU_SHLIBS)
  else
	$(CXX) $(SCI_THIRDPARTY_LIBRARY) $(LDFLAGS) $(SOFLAGS) $(LDRUN_PREFIX)$(LIBDIR_ABS) $(LDRUN_PREFIX)$(SCIRUN_LIBDIR_ABS) -o $@ $(SONAMEFLAG) $(filter %.o,$^) $(patsubst $(SCIRUN_LIBDIR)/lib%.so,-l%,$(filter %.$(SO_OR_A_FILE),$^)) $(REPOSITORIES_$@) $($(notdir $@)_LIBS) $(TAU_MPI_LIBS) $(TAU_SHLIBS)
  endif
  endif

#$(LIBNAME).pure: $(LIBNAME)
#	$(purify) $(CXX) $(SOFLAGS) $(LDRUN_PREFIX)$(LIBDIR_ABS) $(LDRUN_PREFIX)$(SCIRUN_LIBDIR_ABS) -o $@.pure $(SONAMEFLAG) $(filter %.o,$^) $(patsubst $(SCIRUN_LIBDIR)/lib%.so,-l%,$(filter %.so,$^)) $($(notdir $@)_LIBS)

#  These will get removed on make clean
CLEANLIBS := $(CLEANLIBS) $(LIBNAME)
CLEANOBJS := $(CLEANOBJS) $(OBJS)
ifneq ($(REPOSITORY_FLAGS),)
  ALL_LIB_ASSOCIATIONS := $(ALL_LIB_ASSOCIATIONS) $(patsubst %,$(SRCDIR)/ptrepository:%,$(patsubst ./%,%,$(OBJS)))
endif

# Try to prevent user error
SRCS := INVALID_SRCS.cc

ALLOBJS := $(ALLOBJS) $(OBJS)

ifneq ($(GENHDRS),)
ALLGEN := $(ALLGEN) $(GENHDRS)
$(OBJS): $(GENHDRS)
GENHDRS :=
endif
