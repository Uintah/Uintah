#
# Makefile fragment for programs.
#
# $Id$
#

OBJS := $(patsubst %.c,%.o,$(filter %.c,$(SRCS))) \
	   $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS))) \
	   $(patsubst %.s,%.o,$(filter %.s,$(SRCS))) \
	   $(patsubst %.l,%.o,$(filter %.l,$(SRCS))) \
	   $(patsubst %.y,%.o,$(filter %.y,$(SRCS)))

#
# We always link against the internal SCIRun malloc
#
PSELIBS := $(PSELIBS) $(MALLOCLIB)

#
# The libraries are specified like SCICore/Thread but get
# name-mangled to SCICore_Thread
#
PSELIBS := $(subst /,_,$(PSELIBS))

ALLTARGETS := $(ALLTARGETS) $(PROGRAM)

#
# Tuck the value of $(LIBS) away in a mangled variable
#
$(PROGRAM)_LIBS := $(LIBS)

#
# The dependencies can be either .o files or .so files.  The .so
# files are other shared libraries within the PSE.  This allows
# the dependencies between .so's to be expressed conveiently.
#
# There are two complicated substitions here.  The first creates the
# library depdencies by transforming something like SCICore_Thread
# to lib/libSCICore_Thread.so.  The second transforms it from
# lib/libSCICore_Thread.so to -lSCICore_Thread.  This is so that
# we can use the -l syntax to link, but still express the dependicies.
#
$(PROGRAM): $(OBJS) $(patsubst %,$(LIBDIR)lib%.so,$(PSELIBS))
	rm -f $@
	$(CXX) $(LDFLAGS) -o $@ $(filter %.o,$^) $(patsubst ../lib/lib%.so,-l%,$(filter %.so,$^)) $($@_LIBS)

#
#  These will get removed on make clean
#
CLEANOBJS := $(CLEANOBJS) $(OBJS)
CLEANPROGS := $(CLEANPROGS) $(PROGRAM)

#
# Try to prevent user error
#
SRCS := INVALID_SRCS.cc

#
# $Log$
# Revision 1.6  2000/03/17 10:40:17  sparker
# Fixed rule to remove file - use $@ instead of $(PROGRAM)
#
# Revision 1.5  2000/03/17 09:53:22  sparker
# remove before link (bugzilla #39)
#
# Revision 1.4  2000/03/17 09:30:56  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
