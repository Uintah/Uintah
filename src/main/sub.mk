#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := main
SRCS      := $(SRCDIR)/main.cc

ifeq ($(LARGESOS),yes)
  PSELIBS := PSECore SCICore
  ifeq ($(BUILD_PARALLEL),yes)
    PSELIBS := $(PSELIBS) Component
  endif
else
  PSELIBS := PSECore/Dataflow SCICore/Containers SCICore/TclInterface \
	SCICore/Thread SCICore/Exceptions
  ifeq ($(BUILD_PARALLEL),yes)
   PSELIBS := $(PSELIBS) Component/PIDL SCICore/globus_threads
  endif
endif

LIBS := $(GL_LIBS)
ifeq ($(BUILD_PARALLEL),yes)
LIBS := $(LIBS) -L$(GLOBUS_LIB_DIR) -lglobus_io
endif
ifeq ($(NEED_SONAME),yes)
LIBS := $(LIBS) $(XML_LIBRARY) $(TK_LIBRARY) -ldl -lz
endif

PROGRAM := $(PROGRAM_PSE)

CFLAGS_MAIN   := $(CFLAGS) -DPSECORETCL=\"$(SRCTOP_ABS)/PSECore/GUI\" \
                      -DSCICORETCL=\"$(SRCTOP_ABS)/SCICore/GUI\" \
                      -DITCL_WIDGETS=\"$(ITCL_WIDGETS)\" \
                      -DDEFAULT_PACKAGE_PATH=\"$(PACKAGE_PATH)\"

$(SRCDIR)/main.o:	$(SRCDIR)/main.cc Makefile
	$(CXX) $(CFLAGS_MAIN) $(INCLUDES) $(CC_DEPEND_REGEN) -c $< -o $@

include $(SRCTOP)/scripts/program.mk

#
# $Log$
# Revision 1.6  2000/10/20 19:19:31  yarden
# make main.o depend on the toplevel Makefile. This will ensure
# that main.cc is recompiled when a new set of packages is configured
# in.  (main.cc is compiled with PACKAGE_PATH=... flag )
#
# Revision 1.5  2000/06/20 22:36:06  yarden
# add %(GL_LIBS) as the first item while linking pse.
# this is a kludge that enables the Linux version to work
# with NVidia's hardware accelerated drivers (0.9.3).
# the problem showed up as a segmentation error inside _init in
# NVidia GLX library.
#
# Revision 1.4  2000/03/23 10:30:53  sparker
# Now need to link pse with SCICore/Exceptions
#
# Revision 1.3  2000/03/20 21:53:29  yarden
# Linux port: add action for NEED_SONAME
#
# Revision 1.2  2000/03/20 19:39:00  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:48  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
