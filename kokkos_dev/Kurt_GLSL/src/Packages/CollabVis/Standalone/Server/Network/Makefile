# Generated automatically from Makefile.in by configure.
# Makefile for Network

include ../make.include

SRCTOP  := ..
SRCDIR  := .
OBJTOP  := /home/sci/simpson/sv/scirun_upgrade/SCIRun/src/Packages/CollabVis/Standalone/Server

OBJTOP_ABS    := $(shell cd $(OBJTOP) ; pwd)
SRCTOP_ABS    := $(shell cd $(SRCTOP) ; pwd)
ifeq ($(OBJTOP),.)
LIBDIR := lib/
else
LIBDIR := $(OBJTOP_ABS)/lib/
endif
LIBDIR_ABS    := $(shell mkdir -p $(LIBDIR); cd $(LIBDIR) ; pwd)

VPATH := $(SRCDIR)

INCLUDES += -I$(SRCTOP) $(XML_INCLUDE)

INCLUDES := $(sort $(INCLUDES))
#SOFLAGS := $(SOFLAGS) -rpath ${LIBDIR}

OBJS = NetInterface.o NetDispatchManager.o PTPConnection.o \
	MulticastConnection.o NetConnection.o

SOLIB = $(LIBDIR)/libNetwork.so

HELPDIR = AppleSeeds
HELPLIBNAME = libappleseeds.a
HELPLIB = $(HELPDIR)/$(HELPLIBNAME)

LINKLIBS = -L$(XML_LIB_DIR) $(XML_LIB_FLAG)

LIBS = $(LINKLIBS) -L$(LIBDIR) -lNetwork -lThread -lXML -lExceptions \
	-lLogging -lProperties -lCompression -lMessage unpv12e/libunp.a

all:	$(SOLIB) $(HELPLIB)

$(SOLIB) :	$(OBJS) $(HELPLIB)
	$(CXX) $(SOFLAGS) $(OBJS) -o $(SOLIB) $(LINKLIBS) $(LINKER_ARCHIVE) $(HELPLIB)

$(HELPLIB):
	(cd $(HELPDIR) ; $(MAKE) )

clean:	helpclean
	/bin/rm -f $(OBJS) $(SOLIB)

helpclean:
	(cd $(HELPDIR) ; $(MAKE) clean )

# -include $(OBJS:.o=.d)

