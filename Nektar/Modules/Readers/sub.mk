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

SRCDIR   := Nektar/Modules/Readers

SRCS     += \
	$(SRCDIR)/ICNektarReader.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Datatypes \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/Containers SCICore/TclInterface SCICore/Geom
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

