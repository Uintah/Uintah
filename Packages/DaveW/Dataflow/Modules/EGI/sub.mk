#
# Makefile fragment for this subdirectory
# $Id$
#

# *** NOTE ***
# 
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/Modules/EGI

SRCS     += \
	$(SRCDIR)/Anneal.cc\
	$(SRCDIR)/DipoleInSphere.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := PSECore/Datatypes PSECore/Dataflow SCICore/Containers \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/TclInterface SCICore/Datatypes
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.4  2000/06/08 22:46:14  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 20:54:56  moulding
# made changes to allow the module maker to add to and edit this file
#
# Revision 1.2  2000/03/20 19:36:07  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:39  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
