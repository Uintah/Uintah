
SRCDIR := Packages/MIT

SUBDIRS := \
        $(SRCDIR)/Core \
        $(SRCDIR)/Dataflow \
        $(SRCDIR)/StandAlone \

include $(SCIRUN_SCRIPTS)/recurse.mk


