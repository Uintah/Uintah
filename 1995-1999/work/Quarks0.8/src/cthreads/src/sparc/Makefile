# 
# Copyright (c) 1990, 1995 The University of Utah and
# the Computer Systems Laboratory (CSL).  All rights reserved.
#
# Permission to use, copy, modify and distribute this software is hereby
# granted provided that (1) source code retains these copyright, permission,
# and disclaimer notices, and (2) redistributions including binaries
# reproduce the notices in supporting documentation, and (3) all advertising
# materials mentioning features or use of this software display the following
# acknowledgement: ``This product includes software developed by the Computer 
# Systems Laboratory at the University of Utah.''
#
# THE UNIVERSITY OF UTAH AND CSL ALLOW FREE USE OF THIS SOFTWARE IN ITS "AS
# IS" CONDITION.  THE UNIVERSITY OF UTAH AND CSL DISCLAIM ANY LIABILITY OF
# ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
#
# CSL requests users of this software to return to csl-dist@cs.utah.edu any
# improvements that they make and grant CSL redistribution rights.
#
# 	Utah $Hdr$
#
##############################################################################
# File:         Makefile
# Description:  Makefile for machine dependent files of cthreads package.
# Author:       Leigh Stoller
# Created:      16 Nov 90
# RCS $Header$
#
##############################################################################

SOURCE  = stack.c malloc.c preempt.c csw.s lock.s crt0.s
OBJECT  = stack.o malloc.o preempt.o csw.o lock.o crt0.o

CC      = gcc
TOP   = ../..
CFLAGS  =  -g -I$(TOP)/include -DCTHREADS -DSPARC -DPREEMPT


libcthreads_lowlevel.a: $(OBJECT)
	rm -f $@
	ar crv $@ $(OBJECT)
	ranlib $@


crt0.o:	crt0.s
	/lib/cpp -DASSEMBLER $*.s | as -o $*.o -

lock.o:	lock.s
	/lib/cpp -DASSEMBLER $*.s | as -o $*.o -

csw.o:	csw.s
	/lib/cpp -DASSEMBLER $*.s | as -o $*.o -


depend: Makefile $(SOURCE)
	makedepend -fMakefile $(CFLAGS) $(SOURCE)
clean:
	rm -rf a.out *.o

# DO NOT DELETE THIS LINE -- make depend depends on it.

stack.o: ../../include/stack_internal.h ../../include/stack.h
stack.o: ../../include/cthreads.h ../../include/lock.h
stack.o: /usr/include/sys/types.h /usr/include/sys/stdtypes.h
stack.o: /usr/include/sys/sysmacros.h /usr/include/sys/mman.h
stack.o: /usr/include/stdio.h
malloc.o: /usr/include/sys/types.h /usr/include/sys/stdtypes.h
malloc.o: /usr/include/sys/sysmacros.h ../../include/cthreads.h
malloc.o: ../../include/lock.h
preempt.o: /usr/include/stdio.h /usr/include/sys/signal.h
preempt.o: /usr/include/vm/faultcode.h /usr/include/sys/time.h
preempt.o: /usr/include/sys/time.h ../../include/cthreads.h
csw.o: /usr/include/sun4/asm_linkage.h /usr/include/sun4/trap.h
