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

# The libraries are specified like Core/Thread but get
# name-mangled to Core_Thread
PSELIBS := $(subst /,_,$(PSELIBS))

ALLTARGETS := $(ALLTARGETS) $(VOLATILE_LIB)
ALLSRCS := $(ALLSRCS) $(SRCS)

# Tuck the value of $(LIBS) away in a mangled variable
$(VOLATILE_LIB)_LIBS := $(LIBS)

# The dependencies can be either .o files or .so files.  The .so
# files are other shared libraries within the PSE.  This allows
# the dependencies between .so's to be expressed conveiently.
# There are two complicated substitions here.  The first creates the
# library depdencies by transforming something like Core_Thread
# to lib/libCore_Thread.so.  The second transforms it from
# lib/libCore_Thread.so to -lCore_Thread.  This is so that
# we can use the -l syntax to link, but still express the dependicies.
$(VOLATILE_LIB): $(OBJS) $(patsubst %,$(LIBDIR)lib%.so,$(PSELIBS))
	rm -f $@
	$(CXX) $(SOFLAGS) $(LDRUN_PREFIX)$(LIBDIR_ABS) -o $(LIBDIR_ABS)/lib$@.so $(filter %.o,$^) $(patsubst $(LIBDIR)lib%.so,-l%,$(filter %.so,$^)) $($@_LIBS)

#  These will get removed on make clean
CLEANOBJS := $(CLEANOBJS) $(OBJS)
CLEANPROGS := $(CLEANPROGS) $(LIBDIR_ABS)/lib$(VOLATILE_LIB).so

# Try to prevent user error
SRCS := INVALID_SRCS.cc

ifneq ($(GENHDRS),)
ALLGEN := $(ALLGEN) $(GENHDRS)
$(OBJS): $(GENHDRS)
GENHDRS :=
endif

