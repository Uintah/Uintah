#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := PSECore/Dataflow

SRCS     += $(SRCDIR)/Connection.cc $(SRCDIR)/ModuleHelper.cc \
	$(SRCDIR)/Network.cc $(SRCDIR)/Port.cc $(SRCDIR)/Module.cc \
	$(SRCDIR)/NetworkEditor.cc $(SRCDIR)/PackageDB.cc \
        $(SRCDIR)/CreateModulePackage.cc

PSELIBS := PSECore/Comm SCICore/Exceptions SCICore/Thread \
	SCICore/Containers SCICore/TclInterface SCICore/Util \
	SCICore/TkExtensions SCICore/Geom
LIBS := $(TCL_LIBRARY) $(XML_LIBRARY)

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/18 00:17:38  moulding
# Experimental automatic package maker added
#
# Revision 1.1  2000/03/17 09:27:55  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
