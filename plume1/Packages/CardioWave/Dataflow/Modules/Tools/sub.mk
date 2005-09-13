# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

include	$(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR	:= Packages/CardioWave/Dataflow/Modules/Tools

SRCS	+= \
	$(SRCDIR)/PrintF.cc\
	$(SRCDIR)/CreateString.cc\
	$(SRCDIR)/CreateScalar.cc\
	$(SRCDIR)/CreateParametersBundle.cc\
  $(SRCDIR)/FileSelector.cc\
#	$(SRCDIR)/ScalarRange.cc\
#	$(SRCDIR)/SetBundleName.cc\
#	$(SRCDIR)/SetColorMapName.cc\
#	$(SRCDIR)/SetColorMap2Name.cc\
#	$(SRCDIR)/SetFieldName.cc\
#	$(SRCDIR)/SetMatrixName.cc\
#	$(SRCDIR)/SetNrrdName.cc\
#	$(SRCDIR)/SetPathName.cc\

#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/GeomInterface Core/TkExtensions Core/Bundle
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) $(TEEM_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


