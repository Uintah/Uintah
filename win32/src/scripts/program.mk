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
$(PROGRAM): $(OBJS) $(patsubst %,$(LIBDIR)/lib%.$(SO_OR_A_FILE),$(PSELIBS))
	rm -f $@
	$(CXX) $(PROGRAM_LDFLAGS) $(SCI_THIRDPARTY_LIBRARY) $(LDFLAGS) $(LDRUN_PREFIX)$(LIBDIR_ABS) -o $@ $(filter %.o,$^) $(patsubst $(LIBDIR)/lib%.$(SO_OR_A_FILE),-l%,$(filter %.$(SO_OR_A_FILE),$^)) $(REPOSITORIES_$@) $($@_LIBS) $(TAU_LIBRARY)

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

