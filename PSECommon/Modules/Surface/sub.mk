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

SRCDIR   := PSECommon/Modules/Surface

SRCS     += \
	$(SRCDIR)/GenSurface.cc\
	$(SRCDIR)/LabelSurface.cc\
	$(SRCDIR)/SFUGtoSurf.cc\
	$(SRCDIR)/SurfGen.cc\
	$(SRCDIR)/SurfInterpVals.cc\
	$(SRCDIR)/SurfNewVals.cc\
	$(SRCDIR)/SurfToGeom.cc\
	$(SRCDIR)/TransformSurface.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes PSECore/Widgets \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/Containers SCICore/TclInterface SCICore/Geometry \
	SCICore/Datatypes SCICore/Geom
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.5  2000/08/04 19:19:45  dmw
# adding TransformSurface.cc to makefile
#
# Revision 1.4  2000/06/08 22:46:30  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 00:11:40  moulding
# made some modifications that will allow the module make to edit and add
# to this file
#
# Revision 1.2  2000/03/20 19:37:05  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:23  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
