# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

SRCDIR := Dataflow/Modules

SUBDIRS := \
	$(SRCDIR)/DataIO\
	$(SRCDIR)/Fields\
	$(SRCDIR)/Math\
	$(SRCDIR)/Render\
	$(SRCDIR)/SIP\
        $(SRCDIR)/Visualization\
	$(SRCDIR)/ManipFields\
#[INSERT NEW CATEGORY DIR HERE]

include $(SRCTOP)/scripts/recurse.mk

