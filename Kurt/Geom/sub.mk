#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Kurt/Geom

SRCS     += $(SRCDIR)/MultiBrick.cc $(SRCDIR)/VolumeOctree.cc \
	$(SRCDIR)/Brick.cc $(SRCDIR)/TexBrick.cc \
	$(SRCDIR)/VolumeUtils.cc $(SRCDIR)/SliceTable.cc

#
# $Log$
# Revision 1.1  2000/03/17 09:26:31  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#