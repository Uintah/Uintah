
# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/testprograms/PatchBVH

LIBS :=

PROGRAM := $(SRCDIR)/PatchBVHTest
SRCS := $(SRCDIR)/PatchBVHTest.cc

PSELIBS := \
        Core/Exceptions                          \
        Core/Geometry                            \
        Packages/Uintah/Core/Grid                \

LIBS := 

include $(SCIRUN_SCRIPTS)/program.mk

