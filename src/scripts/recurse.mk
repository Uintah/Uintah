# Makefile fragment for this subdirectory

# The SRCDIR_STACK maintains the current state of the recursion.
# This is used to reset SRCDIR after the include below is processed.
SUBDIRS := $(patsubst %,$(SRCTOP)/%,$(SUBDIRS))
SRCDIR_STACK := $(SRCDIR) $(SRCDIR_STACK)
ALLSUBDIRS := $(ALLSUBDIRS) $(SUBDIRS)

TMPSUBDIRS := $(SUBDIRS)
SUBDIRS := SET SUBDIRS BEFORE CALLING RECURSE
include $(patsubst %,%/sub.mk,$(TMPSUBDIRS))

SRCDIR := $(firstword $(SRCDIR_STACK))
SRCDIR_STACK := $(wordlist 2,$(words $(SRCDIR_STACK)),$(SRCDIR_STACK))

