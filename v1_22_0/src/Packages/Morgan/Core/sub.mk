#Makefile fragment for the Packages/Morgan/Core directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Morgan/Core
SUBDIRS := \
	$(SRCDIR)/Dbi\
	$(SRCDIR)/FdStream\
	$(SRCDIR)/Process \
	$(SRCDIR)/PackagePathIterator

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk
