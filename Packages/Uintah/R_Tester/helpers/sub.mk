SRCDIR := Packages/Uintah/R_Tester/helpers

PROGRAM := $(SRCDIR)/compare_dat_files

SRCS	= $(SRCDIR)/compare_dat_files.cc

include $(SCIRUN_SCRIPTS)/program.mk
