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

SRCDIR   := DaveW/Modules/MEG


SRCS     += \
	$(SRCDIR)/EleValuesToMatLabFile.cc\
	$(SRCDIR)/FieldCurl.cc\
	$(SRCDIR)/MakeCurrentDensityField.cc\
	$(SRCDIR)/MagneticFieldAtPoints.cc\
	$(SRCDIR)/MagneticScalarField.cc\
	$(SRCDIR)/NegateGradient.cc\
	$(SRCDIR)/SurfToVectGeom.cc\
	$(SRCDIR)/ForwardMEG.cc \
#[INSERT NEW MODULE HERE]


PSELIBS := DaveW/Datatypes/General PSECore/Dataflow PSECore/Datatypes \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/Datatypes SCICore/TclInterface SCICore/Containers \
	SCICore/Geom
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.6  2000/11/27 22:37:17  vanuiter
# added ForwardMEG
#
# Revision 1.5  2000/10/24 05:57:15  moulding
#
#
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.4  2000/06/08 22:46:17  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 20:54:59  moulding
# made changes to allow the module maker to add to and edit this file
#
# Revision 1.2  2000/03/20 19:36:13  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:51  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
