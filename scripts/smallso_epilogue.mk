# Epilogue fragment for subdirectories.  This is only necessary if
# we are building small sos

ifneq ($(LARGESOS),yes)
include $(SRCTOP)/scripts/so_epilogue.mk
endif

