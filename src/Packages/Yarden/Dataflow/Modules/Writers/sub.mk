#
# Makefile fragment for this subdirectory
# $Id$
#

# *** NOTE ***
# 
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Yarden/Modules/Writers

SRCS     += \
	$(SRCDIR)/TensorFieldWriter.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Yarden/Datatypes PSECore/Datatypes PSECore/Dataflow \
	SCICore/Persistent SCICore/Exceptions SCICore/Containers \
	SCICore/TclInterface SCICore/Thread 
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

