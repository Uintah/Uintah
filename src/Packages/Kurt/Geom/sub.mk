#
# Makefile fragment for this subdirectory
# $Id$
#
include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Kurt/Geom

SRCS     += $(SRCDIR)/MultiBrick.cc $(SRCDIR)/VolumeOctree.cc \
	$(SRCDIR)/Brick.cc $(SRCDIR)/VolumeUtils.cc \
	$(SRCDIR)/SliceTable.cc

PSELIBS := SCICore/Exceptions SCICore/Geometry \
	SCICore/Persistent SCICore/Datatypes \
	SCICore/Containers  SCICore/Geom

LIBS :=  $(LINK) $(GL_LIBS) -lm


include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.4  2000/05/20 02:31:51  kuzimmer
# Multiple changes for new vis tools
#
# Revision 1.3  2000/03/21 17:33:26  kuzimmer
# updating volume renderer
#
# Revision 1.2  2000/03/20 19:36:38  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:31  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
