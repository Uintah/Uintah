TOPDIR= ../..
SRCDIR= $(TOPDIR)/src
INCDIR= $(TOPDIR)/include
LIBDIR= $(TOPDIR)/lib
CC= gcc

TARGET = mp3d
#CCFLAGS = -O2 -DLOCKING
CCFLAGS = -g -I$(INCDIR) -I$(SRCDIR)/cthreads/include -w
#LDFLAGS = -lm
LDFLAGS= -e start -static -nostdlib
LIBS= $(LIBDIR)/libquarks.a $(LIBDIR)/libcthreads.a -lm -lg++ -lgcc -lc

MACROS = ../macros/c.m4.quarks

OBJS = adv.o setup.o mp3d.o

.SUFFIXES:
.SUFFIXES: .o .c .U .h .H
.c.o: ; $(CC) -c $(CCFLAGS) $*.c -o $*.o
.U.c: ; m4 $(MACROS) $*.U >$*.c
.H.h: ; m4 $(MACROS) $*.H >$*.h

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $(TARGET) $(LIBS)

adv.c mp3d.c setup.c: common.h

clean:
	-rm $(TARGET) *.[coh]

