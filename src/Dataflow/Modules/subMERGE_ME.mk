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

SRCDIR := SCIRun/Modules

SUBDIRS := \
	$(SRCDIR)/Image\
	$(SRCDIR)/Mesh\
	$(SRCDIR)/Writers\
#[INSERT NEW CATEGORY DIR HERE]

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.6  2000/12/01 01:34:08  moulding
# TiffWriter requires the TIFF library (go figure).  added #if for TIFF_LIB
# which will presumably be defined by configure one day.
#
# Revision 1.5  2000/10/24 05:57:51  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.4  2000/06/08 22:46:34  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 17:32:58  moulding
# made changes to allow the module maker to add to and edit this file
#
# Revision 1.2  2000/03/20 19:38:10  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:00  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
