
SRCDIR := Packages/DDDAS

SUBDIRS := \
        $(SRCDIR)/Dataflow \
        $(SRCDIR)/StandAlone \

include $(SCIRUN_SCRIPTS)/recurse.mk


