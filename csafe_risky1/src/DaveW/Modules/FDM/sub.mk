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

SRCDIR   := DaveW/Modules/FDM

SRCS     += \
	$(SRCDIR)/BuildFDField.cc\
	$(SRCDIR)/BuildFDMatrix.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := DaveW/Datatypes/General PSECore/Datatypes PSECore/Dataflow \
	SCICore/TclInterface SCICore/Persistent SCICore/Exceptions \
	SCICore/Datatypes SCICore/Thread SCICore/Containers
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.4  2000/06/08 22:46:15  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 20:54:57  moulding
# made changes to allow the module maker to add to and edit this file
#
# Revision 1.2  2000/03/20 19:36:08  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:41  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
