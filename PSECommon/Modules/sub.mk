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

SRCDIR := PSECommon/Modules

SUBDIRS := \
	$(SRCDIR)/Example\
        $(SRCDIR)/FEM\
        $(SRCDIR)/Fields\
	$(SRCDIR)/Iterators\
        $(SRCDIR)/Matrix\
        $(SRCDIR)/Readers\
	$(SRCDIR)/Salmon\
        $(SRCDIR)/Surface\
        $(SRCDIR)/Visualization\
	$(SRCDIR)/Writers\
#[INSERT NEW SUBDIRS HERE]

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.4  2000/06/08 22:46:23  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 00:07:05  moulding
# made modifications to allow module maker to edit and add to this file
#
# Revision 1.2  2000/03/20 19:36:52  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:46  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
