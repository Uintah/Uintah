# Generated automatically from Makefile.in by configure.
# Makefile for XML

include ../../make.include
 
SRCTOP  := ../..
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

INCLUDES += -I$(SRCTOP) $(XML_INCLUDE) $(GL_INCLUDE)

INCLUDES := $(sort $(INCLUDES))

DRIVERS = XMLDriver

SOLIBS =   $(ALL_LIBS)
LIBS = -L$(XML_LIB_DIR) $(XML_LIB_FLAG) -L$(LIBDIR) $(SOLIBS) $(GL_LIB)
 
all: $(DRIVERS)


%:	%.cpp

%:	%.o
	$(CXX) $(CFLAGS) $(INCLUDES) $@.o -o $@ $(LIBS)

test:
	@echo Testing $(shell pwd) - $(shell $(SRCTOP)/runtest $(DRIVERS)) 


clean:
	/bin/rm -f $(DRIVERS) $(DRIVERS:=.o)

# -include $(DRIVERS:=.d) 

# 
# $Log$
# Revision 1.1  2003/07/22 20:59:17  simpson
# Adding CollabVis Client directory to Packages/CollabVis/Standalone
#
# Revision 1.1  2003/06/18 20:55:05  simpson
# Adding CollabVis files/dirs
#
# Revision 1.6  2001/10/10 17:32:43  luke
# Updated config and make to allow for vpath (multi-arch) compilation using the same src tree
#
# Revision 1.5  2001/08/01 19:52:38  luke
# Added malloc, introduced use of malloc into entire code base
#
# Revision 1.4  2001/07/16 20:29:09  luke
# Not modified.
#
# Revision 1.3  2001/05/30 18:45:34  luke
# Added some automated testing
#
# Revision 1.2  2001/05/29 21:35:38  luke
# Fixed rpath problem so we don't need to set LD_LIBRARY_PATH
#
# Revision 1.1  2001/05/29 18:41:54  luke
# Switched to autoconf/configure based system
#
# Revision 1.2  2001/05/21 22:00:45  luke
# Got basic set viewing method to work. Problems still with reading 0 data from net and ignoring outbound messages
#
# Revision 1.1  2001/05/12 03:32:27  luke
# Moved driver to new location
#
# Revision 1.1  2001/05/11 20:10:09  luke
# Initial import of Message drivers
#