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

CSOURCE  = stack.c malloc.c preempt.c 
SSOURCE  = crt0.s  lock.s csw.s
OBJECT  = stack.o malloc.o preempt.o csw.o lock.o crt0.o

AS      = /bin/as
CC      = gcc 
TOP   = ../..
CFLAGS  =  -g -I$(TOP)/include -DCTHREADS -DPREEMPT -D_BSD_SIGNALS -DHPUX -DHPUX700

libcthreads_lowlevel.a: $(OBJECT)
	rm -f $@
	ar crv $@ $(OBJECT)


csw.o: csw.s assym.s

assym.s: genassym
	genassym > assym.s

genassym: genassym.c
	$(CC) -o genassym genassym.c

.depend depend: 
	gcc -MM $(CFLAGS) $(CSOURCE) > .depend

clean:
	rm -rf a.out *.o

.SUFFIXES: .s
.s.o:
	/lib/cpp -DASSEMBLER $< | $(AS)  -o $@

# DO NOT DELETE THIS LINE -- make depend depends on it.

