#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/FEM

SRCS     += \
	$(SRCDIR)/ApplyBC.cc\
	$(SRCDIR)/BuildFEMatrix.cc\
	$(SRCDIR)/ComposeError.cc\
        $(SRCDIR)/ErrorInterval.cc\
	$(SRCDIR)/FEMError.cc\
        $(SRCDIR)/MeshRefiner.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Datatypes \
	SCICore/Persistent SCICore/Thread SCICore/Containers \
	SCICore/Exceptions SCICore/TclInterface SCICore/Geometry
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
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
