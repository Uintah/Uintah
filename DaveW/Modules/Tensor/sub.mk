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

SRCDIR   := DaveW/Modules/Tensor

SRCS     += \
	$(SRCDIR)/Bundles.cc\
	$(SRCDIR)/Flood.cc\
	$(SRCDIR)/TensorAccessFields.cc\
	$(SRCDIR)/TensorAnisotropy.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := DaveW/Datatypes/General PSECore/Datatypes PSECore/Dataflow \
	PSECore/Widgets SCICore/Persistent SCICore/Geometry \
	SCICore/Math SCICore/Exceptions SCICore/Datatypes \
	SCICore/Thread SCICore/Geom SCICore/Containers \
	SCICore/TclInterface 
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.4  2000/06/08 22:46:20  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 20:55:01  moulding
# made changes to allow the module maker to add to and edit this file
#
# Revision 1.2  2000/03/20 19:36:20  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:04  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
