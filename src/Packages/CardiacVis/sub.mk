
SRCDIR := Packages/CardiacVis

SUBDIRS := \
        $(SRCDIR)/Core \
        $(SRCDIR)/Dataflow \

include $(SCIRUN_SCRIPTS)/recurse.mk


