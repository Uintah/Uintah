#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Nektar/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/ICReader.tcl\
#[INSERT NEW TCL FILE HERE]
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/PSECommon/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

