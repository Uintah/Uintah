#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := testprograms/Malloc

PSELIBS :=
LIBS := 

PROGRAM := $(SRCDIR)/test1
SRCS := $(SRCDIR)/test1.cc
include $(OBJTOP_ABS)/scripts/program.mk

PROGRAM := $(SRCDIR)/test2
SRCS := $(SRCDIR)/test2.cc
include $(OBJTOP_ABS)/scripts/program.mk

PROGRAM := $(SRCDIR)/test3
SRCS := $(SRCDIR)/test3.cc
include $(OBJTOP_ABS)/scripts/program.mk

PROGRAM := $(SRCDIR)/test4
SRCS := $(SRCDIR)/test4.cc
include $(OBJTOP_ABS)/scripts/program.mk

PROGRAM := $(SRCDIR)/test5
SRCS := $(SRCDIR)/test5.cc
include $(OBJTOP_ABS)/scripts/program.mk

PROGRAM := $(SRCDIR)/test6
SRCS := $(SRCDIR)/test6.cc
include $(OBJTOP_ABS)/scripts/program.mk

PROGRAM := $(SRCDIR)/test7
SRCS := $(SRCDIR)/test7.cc
include $(OBJTOP_ABS)/scripts/program.mk

PROGRAM := $(SRCDIR)/test8
SRCS := $(SRCDIR)/test8.cc
include $(OBJTOP_ABS)/scripts/program.mk

PROGRAM := $(SRCDIR)/test9
SRCS := $(SRCDIR)/test9.cc
include $(OBJTOP_ABS)/scripts/program.mk

PROGRAM := $(SRCDIR)/test10
SRCS := $(SRCDIR)/test10.cc
include $(OBJTOP_ABS)/scripts/program.mk

PROGRAM := $(SRCDIR)/test11
SRCS := $(SRCDIR)/test11.cc
include $(OBJTOP_ABS)/scripts/program.mk

PROGRAM := $(SRCDIR)/test12
SRCS := $(SRCDIR)/test12.cc
include $(OBJTOP_ABS)/scripts/program.mk

PROGRAM := $(SRCDIR)/test13
SRCS := $(SRCDIR)/test13.cc
include $(OBJTOP_ABS)/scripts/program.mk

PROGRAM := $(SRCDIR)/test14
SRCS := $(SRCDIR)/test14.cc
include $(OBJTOP_ABS)/scripts/program.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:31:14  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#