SRCDIR := Packages/Uintah/Test/helpers

PROGRAM := $(SRCDIR)/compare_dat_files

SRCS	= $(SRCDIR)/compare_dat_files.cc

include $(SRCTOP)/scripts/program.mk
