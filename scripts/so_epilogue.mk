#
# Epilogue fragment for subdirectories.  This is included from
# either smallso_epilogue.mk or largeso_epilogue.mk
#
# $Id$
#

OBJS := $(patsubst %.c,%.o,$(filter %.c,$(SRCS))) \
	   $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS))) \
	   $(patsubst %.s,%.o,$(filter %.s,$(SRCS))) \
	   $(patsubst %.y,%.o,$(filter %.y,$(SRCS)))
LIBNAME := $(LIBDIR)lib$(subst /,_,$(SRCDIR)).so

#
# We always link against the internal SCIRun malloc
#
ifneq ($(SRCDIR),SCICore/Malloc)
ifeq ($(LARGESOS),yes)
else
PSELIBS := $(PSELIBS) $(MALLOCLIB)
endif
endif

#
# The libraries are specified like SCICore/Thread but get
# name-mangled to SCICore_Thread
#
PSELIBS := $(subst /,_,$(PSELIBS))

#
#  These targets will be used to "make all"
#
ALLTARGETS := $(ALLTARGETS) $(LIBNAME)

#
# Tuck the value of $(LIBS) away in a mangled variable
#
$(LIBNAME)_LIBS := $(LIBS)

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
$(LIBNAME): $(OBJS) $(patsubst %,$(LIBDIR)lib%.so,$(PSELIBS))
	rm -f $@
ifeq ($(NEED_SONAME),yes)
	$(CXX) $(SOFLAGS) $(LDRUN_PREFIX)$(LIBDIR_ABS) -o $@ -Wl,-soname,$(notdir $@) $(filter %.o,$^) $(patsubst $(LIBDIR)/lib%.so,-l%,$(filter %.so,$^)) $($@_LIBS)
else
	$(CXX) $(SOFLAGS) $(LDRUN_PREFIX)$(LIBDIR_ABS) -o $@ $(filter %.o,$^) $(patsubst $(LIBDIR)/lib%.so,-l%,$(filter %.so,$^)) $($@_LIBS)
endif
#	$(CXX) $(SOFLAGS) $(LDRUN_PREFIX)$(LIBDIR_ABS) -o $@ -Wl,-soname -Wl,$(notdir $@) $(filter %.o,$^) $(patsubst $(LIBDIR)/lib%.so,-l%,$(filter %.so,$^)) $($@_LIBS)
#
#  These will get removed on make clean
#
CLEANLIBS := $(CLEANLIBS) $(LIBNAME)
CLEANOBJS := $(CLEANOBJS) $(OBJS)

#
# Try to prevent user error
#
SRCS := INVALID_SRCS.cc

ifneq ($(GENHDRS),)
ALLGEN := $(ALLGEN) $(GENHDRS)
$(OBJS): $(GENHDRS)
endif

#
# $Log$
# Revision 1.5  2000/03/20 21:56:22  yarden
# Linux port: add support for so lib on linux
#
# Revision 1.4  2000/03/17 10:40:17  sparker
# Fixed rule to remove file - use $@ instead of $(PROGRAM)
#
# Revision 1.3  2000/03/17 09:53:22  sparker
# remove before link (bugzilla #39)
#
# Revision 1.2  2000/03/17 09:43:45  sparker
# Fixed dependencies for $(GENHDRS)
#
# Revision 1.1  2000/03/17 09:30:58  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
