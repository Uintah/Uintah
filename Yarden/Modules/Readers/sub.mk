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

SRCDIR   := Yarden/Modules/Readers

SRCS     += \
	$(SRCDIR)/TensorFieldReader.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Yarden/Datatypes PSECore/Dataflow PSECore/Datatypes \
	SCICore/Exceptions SCICore/Thread SCICore/Containers \
	SCICore/TclInterface SCICore/Persistent
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk
