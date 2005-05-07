# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/ICE/CustomBCs

SRCS     += $(SRCDIR)/C_BC_driver.cc \
       $(SRCDIR)/NG_NozzleBCs.cc \
       $(SRCDIR)/microSlipBCs.cc \
       $(SRCDIR)/LODI2.cc\
       $(SRCDIR)/MMS_BCs.cc

PSELIBS := \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/Math \
	Core/Exceptions Core/Thread Core/Geometry 

LIBS	:= 


