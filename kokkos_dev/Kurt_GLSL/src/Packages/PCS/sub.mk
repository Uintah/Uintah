
SRCDIR := Packages/PCS

SUBDIRS := \
        $(SRCDIR)/Core \
        $(SRCDIR)/Dataflow \

include $(SCIRUN_SCRIPTS)/recurse.mk


