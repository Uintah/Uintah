#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR	:= Remote/Tools/Model

# grab a copy of $SRCDIR since it changes later
MYSRCDIR := $(SRCTOP)/$(SRCDIR)

$(MYSRCDIR)/BisonMe.cc: $(MYSRCDIR)/BisonMe.y
	bison -v -d $<
	mv -f $(MYSRCDIR)/BisonMe.tab.c $(MYSRCDIR)/BisonMe.cc
	mv -f $(MYSRCDIR)/BisonMe.tab.h $(MYSRCDIR)/BisonMe.h

$(MYSRCDIR)/FlexMe.cc:	$(MYSRCDIR)/FlexMe.l $(MYSRCDIR)/BisonMe.cc
	flex -i -8 -o$@ $< 

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
# Revision 1.3  2000/07/12 18:43:08  dahart
# Removed circular dependency in sub.mk
#
# Revision 1.2  2000/07/11 20:30:10  yarden
# minor bug fixes
#
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
