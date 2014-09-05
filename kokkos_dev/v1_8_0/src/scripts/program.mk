#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

# Makefile fragment for programs.

OBJS := $(patsubst %.c,%.o,$(filter %.c,$(SRCS))) \
	   $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS))) \
	   $(patsubst %.s,%.o,$(filter %.s,$(SRCS))) \
	   $(patsubst %.F,%.o,$(filter %.F,$(SRCS))) \
	   $(patsubst %.l,%.o,$(filter %.l,$(SRCS))) \
	   $(patsubst %.y,%.o,$(filter %.y,$(SRCS)))

# We always link against the internal Dataflow malloc
PSELIBS := $(PSELIBS) $(MALLOCLIB)

ifneq ($(REPOSITORY_FLAGS),)
REPOSITORIES_$(PROGRAM) := $(REPOSITORY_FLAGS) $(SRCDIR)/ptrepository_$(notdir $(PROGRAM)) $(patsubst %,$(REPOSITORY_FLAGS) %/ptrepository, $(PSELIBS))
endif

# The libraries are specified like Core/Thread but get
# name-mangled to Core_Thread
PSELIBS := $(subst /,_,$(PSELIBS))

ALLTARGETS := $(ALLTARGETS) $(PROGRAM)
ALLSRCS := $(ALLSRCS) $(SRCS)

# Tuck the value of $(LIBS) away in a mangled variable
$(PROGRAM)_LIBS := $(LIBS)

# The dependencies can be either .o files or .so files.  The .so
# files are other shared libraries within the PSE.  This allows
# the dependencies between .so's to be expressed conveiently.
# There are two complicated substitions here.  The first creates the
# library depdencies by transforming something like Core_Thread
# to lib/libCore_Thread.so.  The second transforms it from
# lib/libCore_Thread.so to -lCore_Thread.  This is so that
# we can use the -l syntax to link, but still express the dependicies.
$(PROGRAM): $(OBJS) $(patsubst %,$(LIBDIR)lib%.$(SO_OR_A_FILE),$(PSELIBS))
	rm -f $@
	$(CXX) $(LDFLAGS) $(LDRUN_PREFIX)$(LIBDIR_ABS) -o $@ $(filter %.o,$^) $(patsubst $(LIBDIR)lib%.$(SO_OR_A_FILE),-l%,$(filter %.$(SO_OR_A_FILE),$^)) $(REPOSITORIES_$@) $($@_LIBS) $(TAU_MPI_LIBS) $(TAU_SHLIBS)

#  These will get removed on make clean
CLEANOBJS := $(CLEANOBJS) $(OBJS)
CLEANPROGS := $(CLEANPROGS) $(PROGRAM)
ifneq ($(REPOSITORY_FLAGS),)
  ALL_LIB_ASSOCIATIONS := $(ALL_LIB_ASSOCIATIONS) $(patsubst %,$(SRCDIR)/ptrepository_$(notdir $(PROGRAM)):%,$(OBJS))
endif

# Try to prevent user error
SRCS := INVALID_SRCS.cc

ifneq ($(GENHDRS),)
ALLGEN := $(ALLGEN) $(GENHDRS)
$(OBJS): $(GENHDRS)
GENHDRS :=
endif

