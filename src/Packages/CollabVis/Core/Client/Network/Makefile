# Generated automatically from Makefile.in by configure.
# Makefile for Network

include ../make.include

SRCTOP  := ..
SRCDIR  := .
OBJTOP  := /home/sci/simpson/sv/scirun_upgrade/SCIRun/src/Packages/CollabVis/Core/Client

OBJTOP_ABS    := $(shell cd $(OBJTOP) ; pwd)
SRCTOP_ABS    := $(shell cd $(SRCTOP) ; pwd)
ifeq ($(OBJTOP),.)
LIBDIR := lib/
else
LIBDIR := $(OBJTOP_ABS)/lib/
endif
LIBDIR_ABS    := $(shell mkdir -p $(LIBDIR); cd $(LIBDIR) ; pwd)

VPATH := $(SRCDIR)

INCLUDES += -I$(SRCTOP) # $(RMFINCLUDE)

INCLUDES := $(sort $(INCLUDES))
#SOFLAGS := $(SOFLAGS) -rpath ${LIBDIR}

OBJS = PTPConnection.o MulticastConnection.o NetConnection.o \
	NetInterface.o NetDispatchManager.o  

SOLIB = $(LIBDIR)/libNetwork.so


HELPDIR = AppleSeeds
HELPLIBNAME = libappleseeds.a
HELPLIB = $(HELPDIR)/$(HELPLIBNAME)

# RMFDIR = RMF/rmf2.0
# RMFLIBNAME = libRMF.a
# RMFLIB1 = $(RMFDIR)/RMF/src/$(RMFLIBNAME)
# RMFLIB2 = $(RMFDIR)/RMF/usr/local/lib/$(RMFLIBNAME)

LIBS = $(LINKLIBS) -L$(LIBDIR) -lNetwork -lThread -lXML -lExceptions \
	-lLogging -lProperties -lCompression -lMessage 

all:	$(SOLIB) $(HELPLIB) # $(RMFLIB1)

$(SOLIB) :	$(OBJS) $(HELPLIB) # $(RMFLIB1)
	$(CXX) $(SOFLAGS) $(OBJS) -o $(SOLIB) $(LINKER_ARCHIVE) $(HELPLIB) # $(RMFLIB1)

$(HELPLIB):
	(cd $(HELPDIR) ; $(MAKE) )

#$(RMFLIB1):
#	# (cd $(RMFDIR) ; $(MAKE) ; cp $(RMFLIB1) $(RMFLIB2) )


clean:	helpclean
	/bin/rm -f $(OBJS) $(SOLIB)

helpclean::
	(cd $(HELPDIR) ; $(MAKE) clean )
#       (cd $(RMFDIR) ; $(MAKE) clean )

# -include $(OBJS:.o=.d)



