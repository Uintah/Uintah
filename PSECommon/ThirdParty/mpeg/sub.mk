#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := PSECommon/ThirdParty/mpeg

SRCS     += $(SRCDIR)/mpeg.o $(SRCDIR)/codec.o $(SRCDIR)/huffman.o \
	$(SRCDIR)/io.o $(SRCDIR)/chendct.o $(SRCDIR)/lexer.o \
	$(SRCDIR)/marker.o $(SRCDIR)/me.o $(SRCDIR)/mem.o \
	$(SRCDIR)/stat.o $(SRCDIR)/stream.o $(SRCDIR)/transform.o

#
# $Log$
# Revision 1.2  2000/03/20 19:37:11  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:47  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
