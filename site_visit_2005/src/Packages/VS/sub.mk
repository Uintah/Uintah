
SRCDIR := Packages/VS

SUBDIRS := \
        $(SRCDIR)/Core \
        $(SRCDIR)/Dataflow \
        $(SRCDIR)/Standalone \

include $(SCIRUN_SCRIPTS)/recurse.mk


