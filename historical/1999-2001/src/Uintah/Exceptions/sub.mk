#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Exceptions

SRCS     += $(SRCDIR)/InvalidGrid.cc $(SRCDIR)/InvalidValue.cc \
	$(SRCDIR)/ParameterNotFound.cc \
	$(SRCDIR)/ProblemSetupException.cc \
	$(SRCDIR)/TypeMismatchException.cc

PSELIBS := SCICore/Exceptions
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.7  2000/12/23 01:09:01  witzel
# moved UnknownVariable to Uintah/Grid to avoid warnings
#
# Revision 1.6  2000/12/06 23:44:19  witzel
# Added Uintah/Grid to PSELIBS.  This causes a "circular dependency"
# warning while compiling, but it gives many more warnings without it
# because UnknownVariable uses Patch objects explicitly for simplicity
# elsewhere (OnDemandWarehouse and DWDatabase).
#
# Revision 1.5  2000/04/12 22:57:47  sparker
# Added new exception classes
#
# Revision 1.4  2000/04/11 07:10:47  sparker
# Completing initialization and problem setup
# Finishing Exception modifications
#
# Revision 1.3  2000/03/23 10:30:30  sparker
# Update to use new Exception base class
#
# Revision 1.2  2000/03/20 19:38:32  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:54  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
