# Generated automatically from Makefile.in by configure.
# Makefile for Compression

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

OBJS = JPEG.o LZO.o RLE.o ZLIB.o lzo/minilzo.o ppm.o texture.o

SOLIB = ../lib/libCompression.so

LIBS = -ljpeg -lz
DRIVERLIBS = -L$(LIBDIR) -lCompression

all:	$(SOLIB)


$(SOLIB):	$(OBJS)
	$(CXX) $(SOFLAGS)	$(OBJS) -o $(SOLIB) $(LIBS)

clean:
	/bin/rm -f $(OBJS) $(SOLIB)

distclean:	clean
	/bin/rm -f *.ppm *.jpg

# -include $(OBJS:.o=.d)
