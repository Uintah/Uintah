#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := testprograms/Component/pingpong

ifeq ($(LARGESOS),yes)
PSELIBS := SCICore
else
PSELIBS := Component/CIA Component/PIDL SCICore/Thread \
	SCICore/Exceptions SCICore/globus_threads
endif
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common -lglobus_io

PROGRAM := $(SRCDIR)/pingpong
SRCS := $(SRCDIR)/pingpong.cc $(SRCDIR)/PingPong_sidl.cc \
	$(SRCDIR)/PingPong_impl.cc

include $(SRCTOP)/scripts/program.mk

#
# $Log$
# Revision 1.3  2000/03/21 06:13:40  sparker
# Added pattern rule for .sidl files
# Compile component testprograms
#
# Revision 1.2  2000/03/20 19:39:33  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:31:12  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
