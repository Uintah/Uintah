TOPDIR= ../..
SRCDIR= $(TOPDIR)/src
INCDIR= $(TOPDIR)/include
LIBDIR= $(TOPDIR)/lib
CC= gcc
TARGET = WATER
OBJS = water.o redsync.o syscns.o kineti.o bndry.o cnstnt.o initia.o interf.o \
      intraf.o cshift.o poteng.o predcor.o 
CFLAGS= -g -I$(INCDIR) -I$(SRCDIR)/cthreads/include -w
CCFLAGS = $(CFLAGS)
#LDFLAGS = -lm 
LIBS= $(LIBDIR)/libquarks.a $(LIBDIR)/libcthreads.a -lm -lgcc -lc 
LDFLAGS= -e start -static -nostdlib

MACROS = ../macros/c.m4.quarks

.SUFFIXES:
.SUFFIXES: .o .c .U .h .H
.c.o: ; $(CC) -c $(CCFLAGS) $*.c -o $*.o
.U.c: ; m4 $(MACROS) $*.U >$*.c
.H.h: ; m4 $(MACROS) $*.H >$*.h

$(TARGET): $(OBJS) 
	$(CC) $(LDFLAGS) $(OBJS) -o $(TARGET) $(LIBS)

cnstnt.c: cnstnt.U
water.c: water.U  parameters.h mdvar.h water.h wwpot.h cnst.h mddata.h \
         fileio.h frcnst.h randno.h global.h comments.h split.h
redsync.c: redsync.U  parameters.h mdvar.h water.h wwpot.h cnst.h \
          mddata.h fileio.h frcnst.h randno.h global.h
syscns.c: syscns.U
kineti.c: kineti.U global.h
bndry.c: bndry.U
initia.c: initia.U
predcor.c: predcor.U
intraf.c: intraf.U frcnst.h mdvar.h water.h wwpot.h global.h
interf.c: interf.U mdvar.h water.h wwpot.h cnst.h global.h
cshift.c: cshift.U
poteng.c: poteng.U global.h


clean:
	-rm $(TARGET) *.[coh]






