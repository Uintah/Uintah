# Generated automatically from Makefile.in by configure.
# Makefile for XML

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

OBJS = XMLReader.o XMLWriter.o
SOLIB = ../lib/libXML.so

LIBS = -L$(XML_LIB_DIR) $(XML_LIB_FLAG)

all: $(SOLIB)

$(SOLIB) :	$(OBJS)
	$(CXX) $(SOFLAGS)	$(OBJS) -o $(SOLIB) $(LIBS)

clean:
	/bin/rm -f $(OBJS) $(SOLIB)


# -include $(OBJS:.o=.d)

#
# $Log$
# Revision 1.1  2003/07/22 15:46:55  simpson
# Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
#
# Revision 1.1  2003/06/18 22:26:54  simpson
# Adding CollabVis files/dirs
#
# Revision 1.6  2001/10/10 17:32:44  luke
# Updated config and make to allow for vpath (multi-arch) compilation using the same src tree
#
# Revision 1.5  2001/07/16 20:30:14  luke
# Not modified.
#
# Revision 1.4  2001/05/30 18:53:06  luke
# Fixed linker problem that wouldn't allow constants to be properly included in library - this eliminates rpath...
#
# Revision 1.3  2001/05/29 22:34:24  luke
# Linux works again
#
# Revision 1.2  2001/05/29 21:35:39  luke
# Fixed rpath problem so we don't need to set LD_LIBRARY_PATH
#
# Revision 1.1  2001/05/29 18:41:56  luke
# Switched to autoconf/configure based system
#
# Revision 1.4  2001/05/12 03:32:49  luke
# Moved driver to new location
#
