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

SRCDIR   := DaveW/Modules/Readers

SRCS     += \
	$(SRCDIR)/ContourSetReader.cc\
	$(SRCDIR)/SegFldReader.cc\
	$(SRCDIR)/SigmaSetReader.cc\
	$(SRCDIR)/TensorFieldReader.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := DaveW/Datatypes/General PSECore/Dataflow PSECore/Datatypes \
	SCICore/Exceptions SCICore/Thread SCICore/Containers \
	SCICore/TclInterface SCICore/Persistent
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.5  2000/07/19 06:25:13  samsonov
# PathReader moved in PSECommon
#
# Revision 1.4  2000/06/08 22:46:19  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 20:55:01  moulding
# made changes to allow the module maker to add to and edit this file
#
# Revision 1.2  2000/03/20 19:36:18  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:58  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
