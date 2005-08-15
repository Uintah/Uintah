#Makefile fragment for the Packages/RobV directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/RobV
SUBDIRS := \
	$(SRCDIR)/Core \
	$(SRCDIR)/Dataflow \

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk
