# Makefile fragment for this subdirectory

SRCDIR   := tools/sidl
SIDL_SRCDIR := $(SRCDIR)

GENSRCS := $(SRCDIR)/parser.cc $(SRCDIR)/parser.h $(SRCDIR)/scanner.cc
SRCS := $(SRCDIR)/Spec.cc $(SRCDIR)/SymbolTable.cc $(SRCDIR)/emit.cc \
	$(SRCDIR)/main.cc $(SRCDIR)/parser.y $(SRCDIR)/scanner.l \
	$(SRCDIR)/static_check.cc

PROGRAM := $(SRCDIR)/sidl
SIDL_EXE := $(PROGRAM)
PSELIBS :=
LIBS :=

include $(SRCTOP_ABS)/scripts/program.mk

$(SRCDIR)/scanner.cc: $(SRCDIR)/scanner.l
	$(LEX) -t $< > $@

$(SRCDIR)/scanner.o: $(SRCDIR)/parser.h

$(SRCDIR)/parser.cc: $(SRCDIR)/parser.y
	$(YACC) -d $<
	sed 's/getenv()/xxgetenv()/g' < y.tab.c > $@
	rm y.tab.c y.tab.h

$(SRCDIR)/parser.h: $(SRCDIR)/parser.y
	$(YACC) -d $<
	rm y.tab.c
	mv y.tab.h $@

# This dependency is so that the parallel build will not try to build
# parser.cc and parser.h at the same time
$(SRCDIR)/parser.cc: $(SRCDIR)/parser.h

SIDL_BUILTINS := $(SRCTOP_ABS)/Core/CCA/Component/CIA/
CFLAGS_SIDLMAIN   := $(CFLAGS) -DSIDL_BUILTINS=\"$(SIDL_BUILTINS)\"

$(SRCDIR)/main.o:	$(SRCDIR)/main.cc
	$(CXX) $(CFLAGS_SIDLMAIN) $(INCLUDES) $(CC_DEPEND_REGEN) -c $< -o $@

