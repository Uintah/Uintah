# Generated automatically from Makefile.in by configure.
SRCTOP  = ..
SRCDIR  = .
OBJTOP  = ..

TCL_LIB_DIR   = /tmp/opt/itcltk-8.0.4/lib
TCL_LIB_FLAG  = -ltcl8.0
TK_LIB_DIR    = /tmp/opt/itcltk-8.0.4/lib

TK_LIB_FLAG   = -ltk8.0
ITCL_LIB_DIR  = /tmp/opt/itcltk-8.0.4/lib
ITCL_LIB_FLAG = -litk3.0 -litcl3.0

SUBDIRS = GUI Geom Modules
SOLIB   = libKurt.so
ifeq (no,yes)
COMPONENTLIBS = -L$(OBJTOP)/Component -lComponent
GLOBUS_INCLUDE = 
GLOBUS_LIB_DIR = 
GLOBUSLIBS = -L$(GLOBUS_LIB_DIR) -lglobus_nexus -lglobus_io
endif

SOLIBRARIES = -L$(OBJTOP)/PSECore -lPSECore \
	      $(COMPONENTLIBS) \
              -L$(OBJTOP)/SCICore -lSCICore \
	      $(GLOBUSLIBS) \
	      -lGLU -lGL \
              -L$(ITCL_LIB_DIR) $(ITCL_LIB_FLAG) \
              -L$(TK_LIB_DIR) $(TK_LIB_FLAG) \
              -L$(TCL_LIB_DIR) $(TCL_LIB_FLAG) \
              -lm

include $(SRCTOP)/scripts/solib.mk
