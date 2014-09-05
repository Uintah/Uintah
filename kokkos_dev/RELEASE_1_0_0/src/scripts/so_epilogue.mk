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

# Epilogue fragment for subdirectories.  This is included from
# either smallso_epilogue.mk or largeso_epilogue.mk

OBJS := $(patsubst %.c,%.o,$(filter %.c,$(SRCS))) \
	   $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS))) \
	   $(patsubst %.s,%.o,$(filter %.s,$(SRCS))) \
	   $(patsubst %.F,%.o,$(filter %.F,$(SRCS))) \
	   $(patsubst %.y,%.o,$(filter %.y,$(SRCS)))
LIBNAME := $(LIBDIR)lib$(subst /,_,$(SRCDIR)).so

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

#  These targets will be used to "make all"
ALLTARGETS := $(ALLTARGETS) $(LIBNAME)
ALLSRCS := $(ALLSRCS) $(SRCS)

# Tuck the value of $(LIBS) away in a mangled variable
$(LIBNAME)_LIBS := $(LIBS)

# The dependencies can be either .o files or .so files.  The .so
# files are other shared libraries within the PSE.  This allows
# the dependencies between .so's to be expressed conveiently.
# There are two complicated substitions here.  The first creates the
# library depdencies by transforming something like Core_Thread
# to lib/libCore_Thread.so.  The second transforms it from
# lib/libCore_Thread.so to -lCore_Thread.  This is so that
# we can use the -l syntax to link, but still express the dependicies.
ifeq ($(NEED_SONAME),yes)
SONAMEFLAG = -Wl,-soname,$(notdir $@)
else
SONAMEFLAG = 
endif
$(LIBNAME): $(OBJS) $(patsubst %,$(LIBDIR)lib%.so,$(PSELIBS))
	rm -f $@
	$(CXX) $(SOFLAGS) $(LDRUN_PREFIX)$(LIBDIR_ABS) -o $@ $(SONAMEFLAG) $(filter %.o,$^) $(patsubst $(LIBDIR)lib%.so,-l%,$(filter %.so,$^)) $($@_LIBS)

#  These will get removed on make clean
CLEANLIBS := $(CLEANLIBS) $(LIBNAME)
CLEANOBJS := $(CLEANOBJS) $(OBJS)

# Try to prevent user error
SRCS := INVALID_SRCS.cc

ifneq ($(GENHDRS),)
ALLGEN := $(ALLGEN) $(GENHDRS)
$(OBJS): $(GENHDRS)
GENHDRS :=
endif

