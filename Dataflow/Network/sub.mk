#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECore/Dataflow

SRCS     += $(SRCDIR)/Connection.cc $(SRCDIR)/ModuleHelper.cc \
	$(SRCDIR)/Network.cc $(SRCDIR)/Port.cc $(SRCDIR)/Module.cc \
	$(SRCDIR)/NetworkEditor.cc $(SRCDIR)/PackageDB.cc \
	$(SRCDIR)/FileUtils.cc $(SRCDIR)/GenFiles.cc\
	$(SRCDIR)/ComponentNode.cc $(SRCDIR)/SkeletonFiles.cc\
        $(SRCDIR)/PackageDBHandler.cc\
	$(SRCDIR)/StrX.cc

PSELIBS := PSECore/Comm SCICore/Exceptions SCICore/Thread \
	SCICore/Containers SCICore/TclInterface SCICore/Util \
	SCICore/TkExtensions SCICore/Geom PSECore/XMLUtil
LIBS := $(TCL_LIBRARY) $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.7  2000/10/24 05:57:41  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.6  2000/10/21 18:35:10  moulding
# more work for new module maker.
#
# Revision 1.5  2000/10/15 04:17:20  moulding
# Phase 1 for the new module maker: Prepare the repository for the new
# module maker.
#
# added some new files, modified makefiles to include the new files, and
# modified XMLUtil to include some more useful functions.
#
# These changes do not bring the new module maker online.
#
# Revision 1.4  2000/06/07 00:02:32  moulding
# removed the package maker, ane added the module maker
#
# Revision 1.3  2000/03/20 19:37:18  sparker
# Added VPATH support
#
# Revision 1.2  2000/03/18 00:17:38  moulding
# Experimental automatic package maker added
#
# Revision 1.1  2000/03/17 09:27:55  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
