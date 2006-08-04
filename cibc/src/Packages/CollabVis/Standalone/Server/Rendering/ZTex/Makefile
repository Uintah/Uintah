# Generated automatically from Makefile.in by configure.
# Generated automatically from Makefile.in by configure.
# Makefile for Rendering

include ../../make.include

SRCTOP  := ../..
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

OBJS = HeightSimp.o Matrix44.o Object.o SimpMesh.o Image.o Mesh.o \
	Quadric.o


LIB = libZTex.a

all:	$(LIB)

$(LIB) :	$(OBJS)
	$(AR) r	$(LIB) $(OBJS)

clean:
	/bin/rm -f $(OBJS) $(LIB)


# -include $(OBJS:.o=.d)
