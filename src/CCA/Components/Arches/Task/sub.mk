# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/Task

########################################################################
#
# CUDA_ENABLED_SRCS are files that if CUDA is enabled (via configure), must be
# compiled using the nvcc compiler.
#
# WARNING: If you add a file to the list of CUDA_SRCS, you must add a
# corresponding rule at the end of this file!
#
# Also, do not put the .cc on this list of files as the .cc or .cu
# will be added automatically as needed.
#
CUDA_ENABLED_SRCS =         \
        SampleFactory       \
        SampleTask          \
        TaskFactoryBase     \
        TaskInterface       \
        TemplatedSampleTask

ifeq ($(HAVE_CUDA),yes)
   # CUDA enabled files, listed here (and with a rule at the end of
   # this sub.mk) are copied to the binary side and renamed with a .cu
   # extension (.cc replaced with .cu) so that they can be compiled
   # using the nvcc compiler.
   SRCS += $(foreach var,$(CUDA_ENABLED_SRCS),$(OBJTOP_ABS)/$(SRCDIR)/$(var).cu)
   DLINK_FILES := $(DLINK_FILES) $(foreach var,$(CUDA_ENABLED_SRCS),$(SRCDIR)/$(var).o)
else
   SRCS += $(foreach var,$(CUDA_ENABLED_SRCS),$(SRCDIR)/$(var).cc)
endif

########################################################################
# Normal source files:

SRCS += \
        $(SRCDIR)/FieldContainer.cc      

########################################################################
#
# Rules to copy CUDA enabled source (.cc) files to the binary build tree
# and rename with a .cu extension.
#

ifeq ($(HAVE_CUDA),yes)
  # Copy the 'original' .cc files into the binary tree and rename as .cu

  $(OBJTOP_ABS)/$(SRCDIR)/SampleFactory.cu : $(SRCTOP_ABS)/$(SRCDIR)/SampleFactory.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/SampleTask.cu : $(SRCTOP_ABS)/$(SRCDIR)/SampleTask.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/TaskFactoryBase.cu : $(SRCTOP_ABS)/$(SRCDIR)/TaskFactoryBase.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/TaskInterface.cu : $(SRCTOP_ABS)/$(SRCDIR)/TaskInterface.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/TemplatedSampleTask.cu : $(SRCTOP_ABS)/$(SRCDIR)/TemplatedSampleTask.cc
	cp $< $@
endif
