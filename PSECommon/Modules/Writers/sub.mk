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

SRCDIR   := PSECommon/Modules/Writers

SRCS     += \
	$(SRCDIR)/ColorMapWriter.cc\
	$(SRCDIR)/ColumnMatrixWriter.cc\
	$(SRCDIR)/GeometryWriter.cc\
	$(SRCDIR)/MatrixWriter.cc\
	$(SRCDIR)/MeshWriter.cc\
	$(SRCDIR)/MultiScalarFieldWriter.cc\
	$(SRCDIR)/ScalarFieldWriter.cc\
	$(SRCDIR)/SurfaceWriter.cc\
	$(SRCDIR)/TetraWriter.cc\
	$(SRCDIR)/VectorFieldWriter.cc\
	$(SRCDIR)/VoidStarWriter.cc\
	$(SRCDIR)/PathWriter.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Persistent \
	SCICore/Thread SCICore/Exceptions SCICore/TclInterface \
	SCICore/Geom SCICore/Containers SCICore/Datatypes
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.5  2000/07/18 23:14:13  samsonov
# PathWriter module is transfered from DaveW package
#
# Revision 1.4  2000/06/08 22:46:32  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 00:11:42  moulding
# made some modifications that will allow the module make to edit and add
# to this file
#
# Revision 1.2  2000/03/20 19:37:08  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:43  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
