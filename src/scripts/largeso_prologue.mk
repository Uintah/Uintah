# Prologue fragment for subdirectories.  This is only necessary if
# we are building small sos

ifeq ($(LARGESOS),yes)
include $(SRCTOP)/scripts/so_prologue.mk
endif

