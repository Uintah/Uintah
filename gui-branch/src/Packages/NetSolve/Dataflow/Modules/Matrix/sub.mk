# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/NetSolve/Dataflow/Modules/Matrix

SRCS     += \
	$(SRCDIR)/SparseSolve.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network Dataflow/Ports \
	Dataflow/Widgets\
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/TkExtensions \

LIBS := $(TK_LIBRARY) $(GL_LIBS) -L/local/lib -lnetsolve -lm

#/nfs/sci/data1/SCIRun_Thirdparty_32/lib32/libnetsolve.so -lm
#/nfs/sci/data1/SCIRun_Thirdparty_32/lib32/libnetsolve.so -lm
#$(NETSOLVE_ROOT)/lib/SGI64/libnetsolve.a -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


