# Generated automatically from Makefile.in by configure.
# Makefile for logging

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

INCLUDES += -I$(SRCTOP) $(RMFINCLUDE)

INCLUDES := $(sort $(INCLUDES))
#SOFLAGS := $(SOFLAGS) -rpath ${LIBDIR}

OBJS = Util.o

SOLIB = ../lib/libUtil.so


all:	$(SOLIB)

$(SOLIB):	$(OBJS)
	$(CXX) $(SOFLAGS)	$(OBJS) -o $(SOLIB) $(LIBS)

clean:
	/bin/rm -f $(SOLIB) $(OBJS)

# -include $(OBJS:.o=.d)
