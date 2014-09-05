# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Models

SRCS	+= \
       $(SRCDIR)/ModelFactory.cc

SUBDIRS := $(SRCDIR)/test \
           $(SRCDIR)/HEChem
include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS	:= 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
