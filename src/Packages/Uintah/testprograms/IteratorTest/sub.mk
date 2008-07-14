
# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/testprograms/IteratorTest

LIBS :=

PROGRAM := $(SRCDIR)/IteratorTest
SRCS := $(SRCDIR)/IteratorTest.cc

PSELIBS := \
        Core/Exceptions                          \
        Core/Geometry                            \
        Core/Thread                              \
        Core/Util                                \
        Packages/Uintah/Core/Disclosure          \
        Packages/Uintah/Core/Exceptions          \
        Packages/Uintah/Core/Grid                \
        Packages/Uintah/Core/Util                

LIBS := 

include $(SCIRUN_SCRIPTS)/program.mk

