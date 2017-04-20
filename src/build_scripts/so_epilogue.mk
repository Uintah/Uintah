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
# Epilogue fragment for subdirectories.  This is included from
# either smallso_epilogue.mk or largeso_epilogue.mk

OBJS := $(patsubst %.cu,%.$(OBJEXT),$(filter %.cu,$(SRCS))) \
        $(patsubst %.cc,%.$(OBJEXT),$(filter %.cc,$(SRCS)))   \
        $(patsubst %.cpp,%.$(OBJEXT),$(filter %.cpp,$(SRCS))) \
        $(patsubst %.cxx,%.$(OBJEXT),$(filter %.cxx,$(SRCS))) \
        $(patsubst %.c,%.$(OBJEXT),$(filter %.c,$(SRCS)))     \
        $(patsubst %.s,%.$(OBJEXT),$(filter %.s,$(SRCS)))     \
        $(patsubst %.F,%.$(OBJEXT),$(filter %.F,$(SRCS)))     \
        $(patsubst %.f,%.$(OBJEXT),$(filter %.f,$(SRCS)))     \
        $(patsubst %.fif,%.$(OBJEXT),$(filter %.fif,$(SRCS))) \
        $(patsubst %.y,%.$(OBJEXT),$(filter %.y,$(SRCS)))     \
        $(patsubst %.l,%.$(OBJEXT),$(filter %.l,$(SRCS)))

LIBNAME := $(LIBDIR)/lib$(subst /,_,$(SRCDIR)).$(SO_OR_A_FILE)

# We always link against the internal Dataflow malloc
ifneq ($(SRCDIR),Core/Malloc)
  ifeq ($(LARGESOS),yes)
    PSELIBS := $(PSELIBS) $(MALLOCLIB)
  endif
endif

# The libraries are specified like Core/Grid but get
# name-mangled to Core_Grid
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
# library depdencies by transforming something like Core_Grid
# to lib/libCore_Grid.so.  The second transforms it from
# lib/libCore_Grid.so to -lCore_Grid.  This is so that
# we can use the -l syntax to link, but still express the dependicies.
ifeq ($(NEED_SONAME),yes)
  SONAMEFLAG = -Wl,-soname -Wl,$(notdir $@)
else
  SONAMEFLAG = 
endif

TMPPSELIBS = $(patsubst %,lib%.$(SO_OR_A_FILE),$(PSELIBS)) 
TMPP = $(patsubst libPackages_%,PACKAGE%,$(TMPPSELIBS))
TMPLIBS = $(patsubst lib%,SCIRUN%,$(TMPP))
TMP_CORE_PSELIBS = $(patsubst SCIRUN%,lib%,$(TMPLIBS))
CORE_PSELIBS = $(patsubst PACKAGE%,,$(TMP_CORE_PSELIBS))
TMP_PACK_PSELIBS = $(patsubst PACKAGE%,libPackages_%,$(TMPLIBS))
PACK_PSELIBS = $(patsubst SCIRUN%,,$(TMP_PACK_PSELIBS))
COMMON_LIBS = ${MALLOC_TRACE_LIB_DIR_FLAG} ${MALLOC_TRACE_LIB_FLAG} 

$(LIBNAME): $(OBJS) $(patsubst %,$(SCIRUN_LIBDIR)/%,$(CORE_PSELIBS)) $(patsubst %,$(LIBDIR)/%,$(PACK_PSELIBS))
  ifeq ($(SCI_MAKE_BE_QUIET),true)
	@rm -f $@
  else
	rm -f $@
  endif
  ifeq ($(MAKE_ARCHIVES),yes)
    ifeq ($(SCI_MAKE_BE_QUIET),true)
	@echo "Creating Archive:   $@"
	@ar -q $@ $(filter %.$(OBJEXT),$^)  2> /dev/null
    else
	ar -v -q $@ $(filter %.$(OBJEXT),$^)
    endif
  else
    ifeq ($(IS_OSX),yes)
      ifeq ($(SCI_MAKE_BE_QUIET),true)
	@echo "Linking:   $@"
	@$(CXX) $(SCI_THIRDPARTY_LIBRARY) -single_module $(LDFLAGS) $(SOFLAGS) -install_name $(SCIRUN_LIBDIR_ABS)/$(patsubst lib/%,%,$@) -o $@ $(SONAMEFLAG) $(filter %.$(OBJEXT),$^) $(patsubst $(SCIRUN_LIBDIR)/lib%.so,-l%,$(filter %.$(SO_OR_A_FILE),$^)) $(REPOSITORIES_$@) $($(notdir $@)_LIBS) ${COMMON_LIBS}
      else
	$(CXX) $(SCI_THIRDPARTY_LIBRARY) -single_module $(LDFLAGS) $(SOFLAGS) -install_name $(SCIRUN_LIBDIR_ABS)/$(patsubst lib/%,%,$@) -o $@ $(SONAMEFLAG) $(filter %.$(OBJEXT),$^) $(patsubst $(SCIRUN_LIBDIR)/lib%.so,-l%,$(filter %.$(SO_OR_A_FILE),$^)) $(REPOSITORIES_$@) $($(notdir $@)_LIBS) ${COMMON_LIBS}
      endif
    else
      ifeq ($(SCI_MAKE_BE_QUIET),true)
	    @echo "Linking:   $@"
	    @$(CXX) $(SCI_THIRDPARTY_LIBRARY) $(LDFLAGS) $(SOFLAGS) $(LDRUN_PREFIX)$(LIBDIR_ABS) $(LDRUN_PREFIX)$(SCIRUN_LIBDIR_ABS) -o $@ $(SONAMEFLAG) $(filter %.$(OBJEXT),$^) $(patsubst $(SCIRUN_LIBDIR)/lib%.so,-l%,$(filter %.$(SO_OR_A_FILE),$^)) $(REPOSITORIES_$@) $($(notdir $@)_LIBS) ${COMMON_LIBS}
      else
	    $(CXX) $(SCI_THIRDPARTY_LIBRARY) $(LDFLAGS) $(SOFLAGS) $(LDRUN_PREFIX)$(LIBDIR_ABS) $(LDRUN_PREFIX)$(SCIRUN_LIBDIR_ABS) -o $@ $(SONAMEFLAG) $(filter %.$(OBJEXT),$^) $(patsubst $(SCIRUN_LIBDIR)/lib%.so,-l%,$(filter %.$(SO_OR_A_FILE),$^)) $(REPOSITORIES_$@) $($(notdir $@)_LIBS) ${COMMON_LIBS}
      endif
    endif
  endif

#$(LIBNAME).pure: $(LIBNAME)
#	$(purify) $(CXX) $(SOFLAGS) $(LDRUN_PREFIX)$(LIBDIR_ABS) $(LDRUN_PREFIX)$(SCIRUN_LIBDIR_ABS) -o $@.pure $(SONAMEFLAG) $(filter %.$(OBJEXT),$^) $(patsubst $(SCIRUN_LIBDIR)/lib%.so,-l%,$(filter %.so,$^)) $($(notdir $@)_LIBS)

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
