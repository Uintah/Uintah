# Makefile fragment for this subdirectory

SRCDIR   := main
SRCS      := $(SRCDIR)/main.cc

ifeq ($(LARGESOS),yes)
  PSELIBS := Dataflow Core
  ifeq ($(BUILD_PARALLEL),yes)
    PSELIBS := $(PSELIBS) Core/CCA/Component
  endif
else
  PSELIBS := Dataflow/Network Core/Containers Core/TclInterface \
	Core/Thread Core/Exceptions
  ifeq ($(BUILD_PARALLEL),yes)
   PSELIBS := $(PSELIBS) Component/PIDL Core/globus_threads
  endif
endif

LIBS := $(GL_LIBS)
ifeq ($(BUILD_PARALLEL),yes)
LIBS := $(LIBS) -L$(GLOBUS_LIB_DIR) -lglobus_io
endif
ifeq ($(NEED_SONAME),yes)
LIBS := $(LIBS) $(XML_LIBRARY) $(TK_LIBRARY) -ldl -lz
endif

PROGRAM := $(PROGRAM_PSE)

CFLAGS_MAIN   := $(CFLAGS) -DPSECORETCL=\"$(SRCTOP_ABS)/Dataflow/GUI\" \
                      -DSCICORETCL=\"$(SRCTOP_ABS)/Core/GUI\" \
                      -DITCL_WIDGETS=\"$(ITCL_WIDGETS)\" \
                      -DDEFAULT_PACKAGE_PATH=\"$(PACKAGE_PATH)\"

$(SRCDIR)/main.o:	$(SRCDIR)/main.cc Makefile
	$(CXX) $(CFLAGS_MAIN) $(INCLUDES) $(CC_DEPEND_REGEN) -c $< -o $@

include $(SRCTOP)/scripts/program.mk

