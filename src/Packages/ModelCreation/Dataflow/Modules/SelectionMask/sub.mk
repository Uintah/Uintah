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

SRCDIR   := Packages/ModelCreation/Dataflow/Modules/SelectionMask

SRCS     += \
  	$(SRCDIR)/SelectionMaskAND.cc\
  	$(SRCDIR)/SelectionMaskOR.cc\
  	$(SRCDIR)/SelectionMaskXOR.cc\
  	$(SRCDIR)/SelectionMaskNOT.cc\
  	$(SRCDIR)/SelectionMaskToIndices.cc\
  	$(SRCDIR)/IndicesToSelectionMask.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Dataflow/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/GeomInterface Dataflow/TkExtensions \
        Packages/ModelCreation/Core/Fields 
        
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


