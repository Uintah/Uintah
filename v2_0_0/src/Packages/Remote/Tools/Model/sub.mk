# Makefile fragment for this subdirectory

SRCDIR	:= Packages/Remote/Tools/Model

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

