# Generated automatically from Makefile.in by configure.
#
# Default target
#
default: all


#
# Directory variables
#
SRCTOP  := .
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

VPATH := $(SRCTOP)

#
# Subdirectories
# 
SUBDIRS := Compression Exceptions Logging Message Network Properties\
	Rendering Thread Malloc XML UI Drivers

#
# Libraries and other flags
#
THREAD_LIBS := -lpthread
PLATFORM_LDFLAGS := -ldl
TRACEBACK_LIB := 

# XML
XML_LIB_DIR := /home/sci/simpson/sv/scirun_upgrade/SCIRun/linux/SCIRun_Thirdparty/1.10/Linux/gcc-3.2.2-32bit/src/xerces-c-src2_1_0/lib/
XML_LIB_FLAG := -lxerces-c
XML_LIBRARY := $(LDRUN_PREFIX)$(XML_LIB_DIR) -L$(XML_LIB_DIR) $(XML_LIB_FLAG)
XML_INCLUDE := -I/home/sci/simpson/sv/scirun_upgrade/SCIRun/linux/SCIRun_Thirdparty/1.10/Linux/gcc-3.2.2-32bit/src/xerces-c-src2_1_0/include/

INCLUDES += -I$(SRCTOP)

INCLUDES := $(sort $(INCLUDES))

# variables and suffix rules for module transformation
#------------------------------------------------------------------------------

CC              := gcc
CXX             := c++
AS              := 
LD              := 
CFLAGS          := -Wall -g -O2 $(CFLAGS)
SOFLAGS         := -shared  -L$(LIBDIR) $(CFLAGS)
LDFLAGS         :=  -L$(LIBDIR) $(CFLAGS)
ASFLAGS         :=   $(ASFLAGS)
CC_DEPEND_REGEN = -MD
NEED_SONAME	:= yes
F77		:= 
FFLAGS		:=  $(FFLAGS)
F77_DEPEND_REGEN = -MD
ifeq ($(NEED_SONAME),yes)
SONAMEFLAG = -Wl,-soname,$(notdir $@)
else
SONAMEFLAG = 
endif


.SUFFIXES: .cc .c .cpp

.cc.o: $<
	$(CXX) $(CFLAGS) $(INCLUDES) $(CC_DEPEND_REGEN) -c $< -o $@
ifeq ($(CC_DEPEND_REGEN),-MD)
	mv $(basename $(notdir $@)).d $(basename $@).d
endif

.c.o: $<
	$(CC) $(CFLAGS) $(INCLUDES) $(CC_DEPEND_REGEN) -c $< -o $@
ifeq ($(CC_DEPEND_REGEN),-MD)
	mv $(basename $(notdir $@)).d $(basename $@).d
endif

.cpp.o: $<
	$(CXX) $(CFLAGS) $(INCLUDES) $(CC_DEPEND_REGEN) -c $< -o $@
ifeq ($(CC_DEPEND_REGEN),-MD)
	mv $(basename $(notdir $@)).d $(basename $@).d
endif

all:: 
	( cd Compression ; $(MAKE) $(MAKEFLAGS) ) 
	( cd Logging ; $(MAKE) $(MAKEFLAGS) ) 
	( cd XML ; $(MAKE) $(MAKEFLAGS) ) 
	( cd Thread ; $(MAKE) $(MAKEFLAGS) ) 
	( cd Exceptions ; $(MAKE) $(MAKEFLAGS) ) 
	( cd Malloc ; $(MAKE) $(MAKEFLAGS) ) 
	( cd Message ; $(MAKE) $(MAKEFLAGS) ) 
	( cd Network ; $(MAKE) $(MAKEFLAGS) ) 
	( cd UI ; $(MAKE) $(MAKEFLAGS) ) 
	( cd Util ; $(MAKE) $(MAKEFLAGS) ) 
	( cd Rendering ; $(MAKE) $(MAKEFLAGS) )  
	( cd Drivers ; $(MAKE) $(MAKEFLAGS) ) 

clean::	
	( cd Compression ; $(MAKE) clean ) 
	( cd Logging ; $(MAKE) clean )
	( cd XML ; $(MAKE) clean ) 
	( cd Message ; $(MAKE) clean )
	( cd Thread ; $(MAKE) clean )
	( cd Exceptions ; $(MAKE) clean )
	( cd Malloc ; $(MAKE) clean )
	( cd Network ; $(MAKE) clean ) 	
	( cd UI ; $(MAKE) clean ) 
	( cd Util ; $(MAKE) clean ) 
	( cd Rendering ; $(MAKE) clean )  
	( cd Drivers ; $(MAKE) clean ) 

distclean:	clean
	find . -name config.cache -exec rm {} \; 	
	find . -name config.status -exec rm {} \;
	find . -name '*.log' -exec rm {} \;
	find . -name '*~' -exec rm {} \;
	find . -name make.include -exec rm {} \;
ifeq ($(CC_DEPEND_REGEN),-MD)
	find . -name '*.d' -exec rm {} \;
else	
	find . -name 'depend.mk' -exec rm {} \;
endif
	find . -name 'ii_files' -exec /bin/rm -rf {} \;
