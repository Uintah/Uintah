#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Remote/Modules/remoteSalmon

SRCS     += $(SRCDIR)/remoteSalmon.cc \
	$(SRCDIR)/Image.cc $(SRCDIR)/Matrix44.cc \
	$(SRCDIR)/HeightSimp.cc $(SRCDIR)/Object.cc $(SRCDIR)/Utils.cc \
	$(SRCDIR)/RenderModel.cc $(SRCDIR)/Mesh.cc $(SRCDIR)/Quadric.cc \
	$(SRCDIR)/SimpMesh.cc $(SRCDIR)/OpenGLServer.cc \
	$(SRCDIR)/socketServer.cc

PSELIBS := PSECore/Dataflow PSECore/Datatypes PSECore/Comm \
	SCICore/Persistent SCICore/Exceptions SCICore/Geometry \
	SCICore/Geom SCICore/Thread SCICore/Containers \
	SCICore/TclInterface SCICore/TkExtensions SCICore/Util \
	SCICore/TkExtensions SCICore/Datatypes \
	PSECommon/Modules/Salmon SCICore/OS 

LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/06/06 15:29:31  dahart
# - Added a new package / directory tree for the remote visualization
# framework.
# - Added a new Salmon-derived module with a small server-daemon,
# network support, and a couple of very basic remote vis. services.
#
# Revision 1.1  2000/03/17 09:27:18  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
