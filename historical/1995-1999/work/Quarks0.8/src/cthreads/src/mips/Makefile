##############################################################################
# File:         Makefile
# Description:  Makefile for machine dependent files of cthreads package.
# Author:       Leigh Stoller
# Created:      16 Nov 90
# RCS $Header$
#
# (c) Copyright 1990, University of Utah, all rights reserved.
##############################################################################

SOURCE  = stack.c malloc.c preempt.c csw.s lock.c Main.c
OBJECT  = stack.o malloc.o preempt.o csw.o lock.o crt0.o Main.o

CC      = cc
TOP   = ../..
CFLAGS  =  -g -I$(TOP)/include -DCTHREADS -DSGI -DPREEMPT -D_BSD_SIGNALS
AFLAGS  = -DASSEMBLER

libcthreads_lowlevel.a: $(OBJECT)
	rm -f $@
	ar crv $@ $(OBJECT)


Main.o: Main.c
	gcc -c Main.c

csw.o:	csw.s
	as $(AFLAGS) -o $*.o $*.s


depend: Makefile $(SOURCE)
	makedepend -fMakefile $(CFLAGS) $(SOURCE)
clean:
	rm -rf a.out *.o

# DO NOT DELETE THIS LINE -- make depend depends on it.

stack.o: ../../include/stack_internal.h ../../include/stack.h
stack.o: ../../include/cthreads.h ../../include/lock.h
stack.o: /usr/include/sys/mman.h /usr/include/sys/types.h
stack.o: /usr/include/sgidefs.h /usr/include/stdio.h
malloc.o: /usr/include/sys/types.h /usr/include/sgidefs.h
malloc.o: ../../include/cthreads.h ../../include/lock.h
preempt.o: /usr/include/stdio.h /usr/include/sys/signal.h
preempt.o: /usr/include/sys/time.h ../../include/cthreads.h
csw.o: /usr/include/regdef.h /usr/include/sys/regdef.h /usr/include/sgidefs.h
csw.o: /usr/include/sys/fpregdef.h /usr/include/asm.h /usr/include/sys/asm.h
lock.o: /usr/include/assert.h ../../include/lock.h
crt0.o: /usr/include/regdef.h /usr/include/sys/regdef.h
crt0.o: /usr/include/sgidefs.h /usr/include/sys/fpregdef.h /usr/include/asm.h
crt0.o: /usr/include/sys/asm.h
