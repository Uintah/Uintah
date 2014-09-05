include $(OBJTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Moulding

SUBDIRS := $(SRCDIR)/GUI $(SRCDIR)/Datatypes \
        $(SRCDIR)/Modules

include $(OBJTOP_ABS)/scripts/recurse.mk

PSELIBS := PSECore SCICore
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/largeso_epilogue.mk


