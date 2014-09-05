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

SRCDIR   := PSECommon/Modules/FEM

SRCS     += \
	$(SRCDIR)/ApplyBC.cc\
	$(SRCDIR)/BuildFEMatrix.cc\
	$(SRCDIR)/ComposeError.cc\
        $(SRCDIR)/ErrorInterval.cc\
	$(SRCDIR)/FEMError.cc\
        $(SRCDIR)/MeshRefiner.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Datatypes \
	SCICore/Persistent SCICore/Thread SCICore/Containers \
	SCICore/Exceptions SCICore/TclInterface SCICore/Geometry
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2.2.4  2000/11/01 23:02:33  mcole
# Fix for previous merge from trunk
#
# Revision 1.2.2.2  2000/10/26 10:03:29  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.2.2.1  2000/09/28 03:16:51  mcole
# merge trunk into FIELD_REDESIGN branch
#
# Revision 1.5  2000/10/24 05:57:32  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.4  2000/06/08 22:46:25  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 00:11:36  moulding
# made some modifications that will allow the module make to edit and add
# to this file
#
# Revision 1.2  2000/03/20 19:36:55  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:53  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
