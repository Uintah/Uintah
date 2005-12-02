# Generated automatically from Makefile.in by configure.
# Makefile for Network

include ../make.include

SRCTOP  := ..
SRCDIR  := .
OBJTOP  := /home/sci/simpson/sv/scirun_upgrade/SCIRun/src/Packages/CollabVis/Standalone/Client

OBJTOP_ABS    := $(shell cd $(OBJTOP) ; pwd)
SRCTOP_ABS    := $(shell cd $(SRCTOP) ; pwd)
ifeq ($(OBJTOP),.)
LIBDIR := lib/
else
LIBDIR := $(OBJTOP_ABS)/lib/
endif
LIBDIR_ABS    := $(shell mkdir -p $(LIBDIR); cd $(LIBDIR) ; pwd)

VPATH := $(SRCDIR)

INCLUDES += -I$(SRCTOP) 

INCLUDES := $(sort $(INCLUDES))
#SOFLAGS := $(SOFLAGS) -rpath ${LIBDIR}

OBJS = PTPConnection.o MulticastConnection.o NetConnection.o \
	NetInterface.o NetDispatchManager.o  

SOLIB = $(LIBDIR)/libNetwork.so


HELPDIR = AppleSeeds
HELPLIBNAME = libappleseeds.a
HELPLIB = $(HELPDIR)/$(HELPLIBNAME)

LIBS = $(LINKLIBS) -L$(LIBDIR) -lNetwork -lThread -lXML -lExceptions \
	-lLogging -lProperties -lCompression -lMessage 

all:	$(SOLIB) $(HELPLIB) 

$(SOLIB) :	$(OBJS) $(HELPLIB) 
	$(CXX) $(SOFLAGS) $(OBJS) -o $(SOLIB) $(LINKER_ARCHIVE) $(HELPLIB) 

$(HELPLIB):
	(cd $(HELPDIR) ; $(MAKE) )


clean:	helpclean
	/bin/rm -f $(OBJS) $(SOLIB)

helpclean::
	(cd $(HELPDIR) ; $(MAKE) clean )

# -include $(OBJS:.o=.d)



