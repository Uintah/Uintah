# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/CCA/Component/CIA

SRCS     += \
	$(SRCDIR)/CIA_sidl.cc \
	$(SRCDIR)/Class.cc \
	$(SRCDIR)/ClassNotFoundException.cc \
	$(SRCDIR)/IllegalArgumentException.cc \
	$(SRCDIR)/InstantiationException.cc \
	$(SRCDIR)/Interface.cc \
	$(SRCDIR)/Method.cc \
	$(SRCDIR)/NoSuchMethodException.cc \
	$(SRCDIR)/Object.cc \
	$(SRCDIR)/Throwable.cc 

# We cannot use the implicit rule for CIA, since it needs that
# special -cia flag
$(SRCDIR)/CIA_sidl.o: $(SRCDIR)/CIA_sidl.cc $(SRCDIR)/CIA_sidl.h

$(SRCDIR)/CIA_sidl.cc: $(SRCDIR)/CIA.sidl $(SIDL_EXE)
	$(SIDL_EXE) -cia -o $@ $<

$(SRCDIR)/CIA_sidl.h: $(SRCDIR)/CIA.sidl $(SIDL_EXE)
	$(SIDL_EXE) -cia -h -o $@ $<

GENHDRS := $(SRCDIR)/CIA_sidl.h

PSELIBS := Core/CCA/Component/PIDL
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common

include $(SRCTOP)/scripts/smallso_epilogue.mk

