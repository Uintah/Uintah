#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Component/CIA

SRCS     += $(SRCDIR)/CIA_sidl.cc $(SRCDIR)/Class.cc \
	$(SRCDIR)/ClassNotFoundException.cc \
	$(SRCDIR)/IllegalArgumentException.cc \
	$(SRCDIR)/InstantiationException.cc \
	$(SRCDIR)/Interface.cc $(SRCDIR)/Method.cc \
	$(SRCDIR)/NoSuchMethodException.cc $(SRCDIR)/Object.cc \
	$(SRCDIR)/Throwable.cc 

#
# We cannot use the implicit rule for CIA, since it needs that
# special -cia flag
$(SRCDIR)/CIA_sidl.o: $(SRCDIR)/CIA_sidl.cc $(SRCDIR)/CIA_sidl.h

$(SRCDIR)/CIA_sidl.cc: $(SRCDIR)/CIA.sidl $(SIDL_EXE)
	$(SIDL_EXE) -cia -o $@ $<

$(SRCDIR)/CIA_sidl.h: $(SRCDIR)/CIA.sidl $(SIDL_EXE)
	$(SIDL_EXE) -cia -h -o $@ $<

GENHDRS := $(SRCDIR)/CIA_sidl.h

PSELIBS := Component/PIDL
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.5  2000/03/23 10:55:02  sparker
# Put back rule for creating CIA_sidl.h, since it requires the -cia flag
#   for the sidl compiler
#
# Revision 1.4  2000/03/21 06:13:31  sparker
# Added pattern rule for .sidl files
# Compile component testprograms
#
# Revision 1.3  2000/03/20 19:35:45  sparker
# Added VPATH support
#
# Revision 1.2  2000/03/17 09:43:44  sparker
# Fixed dependencies for $(GENHDRS)
#
# Revision 1.1  2000/03/17 09:25:11  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
