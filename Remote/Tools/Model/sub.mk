#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR	:= Remote/Tools/Model

	# grab a copy of $SRCDIR since it changes later
MYSRCDIR := $(SRCDIR)

$(MYSRCDIR)/BisonMe.cc: $(MYSRCDIR)/BisonMe.y $(MYSRCDIR)/FlexMe.cc
	bison -v -d $(MYSRCDIR)/BisonMe.y
	mv -f $(MYSRCDIR)/BisonMe.tab.c $(MYSRCDIR)/BisonMe.cc
	mv -f $(MYSRCDIR)/BisonMe.tab.h $(MYSRCDIR)/BisonMe.h

$(MYSRCDIR)/FlexMe.cc:	$(MYSRCDIR)/FlexMe.l $(MYSRCDIR)/BisonMe.cc
	flex -i -8 -o$(MYSRCDIR)/FlexMe.cc $(MYSRCDIR)/FlexMe.l 

SRCS     += \
	$(SRCDIR)/BisonMe.cc \
	$(SRCDIR)/Mesh.cc \
	$(SRCDIR)/ReadOBJ.cc \
	$(SRCDIR)/SaveOBJ.cc \
	$(SRCDIR)/Object.cc \
	$(SRCDIR)/ReadVRML.cc \
	$(SRCDIR)/SaveVRML.cc

#	$(SRCDIR)/FlexMe.cc \

#
# $Log$
# Revision 1.1  2000/07/10 20:44:22  dahart
# initial commit
#
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
