#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := testprograms/Component/argtest

ifeq ($(LARGESOS),yes)
PSELIBS := SCICore
else
PSELIBS := Component/CIA Component/PIDL SCICore/Thread \
	SCICore/Exceptions SCICore/globus_threads
endif
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common -lglobus_io

PROGRAM := $(SRCDIR)/argtest
SRCS := $(SRCDIR)/argtest.cc $(SRCDIR)/argtest_sidl.cc
GENHDRS := $(SRCDIR)/argtest_sidl.h

include $(SRCTOP)/scripts/program.mk

#
# $Log$
# Revision 1.4  2000/03/23 11:05:19  sparker
# Added *_sidl.h files to GENHDRS so that they will get built in time
#
# Revision 1.3  2000/03/21 06:13:34  sparker
# Added pattern rule for .sidl files
# Compile component testprograms
#
# Revision 1.2  2000/03/20 19:39:18  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:31:04  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
