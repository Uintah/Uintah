# Generated automatically from Makefile.in by configure.
# Makefile for exceptions

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

OBJS = Exception.o

SOLIB = ../lib/libExceptions.so

LIBS = 


all:	$(SOLIB)

$(SOLIB):	$(OBJS)
	$(CXX) $(SOFLAGS)	$(OBJS) -o $(SOLIB) $(LIBS)

clean:
	/bin/rm -f $(SOLIB) $(OBJS)

# -include $(OBJS:.o=.d)
