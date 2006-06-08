# Generated automatically from Makefile.in by configure.
# Makefile for XML

include ../../make.include
 
#
# Directory variables
#
SRCTOP  := ../..
SRCDIR  := .
OBJTOP  := /home/sci/simpson/sv/scirun_upgrade/SCIRun/src/Packages/CollabVis/Standalone/Client

OBJTOP_ABS    := $(shell cd $(OBJTOP) ; pwd)
SRCTOP_ABS    := $(shell cd $(SRCTOP) ; pwd)
ifeq ($(OBJTOP),.)
LIBDIR := lib/
else
LIBDIR := $(OBJTOP_ABS)/lib/
endif
LIBDIR_ABS    := $(shell mkdir -p $(LIBDIR); cd $(LIBDIR) ; pwd)

VPATH := $(SRCDIR)

INCLUDES += -I$(SRCTOP) $(XML_INCLUDE)

INCLUDES := $(sort $(INCLUDES))

DRIVERS = UIDriver

SOLIBS = $(ALL_LIBS)
LIBS = -L$(XML_LIB_DIR) $(XML_LIB_FLAG) -L$(LIBDIR) $(SOLIBS)

all: $(DRIVERS)

%:	%.cpp

%:	%.o
	$(CXX) $(CFLAGS) $(INCLUDES) $@.o -o $@ $(LIBS)

test:
	@echo Testing $(shell pwd) - $(shell $(SRCTOP)/runtest $(DRIVERS)) 

clean:
	/bin/rm -f $(DRIVERS) $(DRIVERS:=.o)

# -include $(DRIVERS:=.d) 


