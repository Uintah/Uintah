# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Fields

SRCS     += \
	$(SRCDIR)/ApplyInterpolant.cc\
	$(SRCDIR)/BuildInterpolant.cc\
	$(SRCDIR)/CastField.cc\
	$(SRCDIR)/FieldBoundary.cc\
	$(SRCDIR)/ManageFieldSet.cc\
	$(SRCDIR)/SeedField.cc\
	$(SRCDIR)/SetFieldData.cc\
	$(SRCDIR)/TransformField.cc\
#[INSERT NEW CODE FILE HERE]
#	$(SRCDIR)/ChangeCellType.cc\
#	$(SRCDIR)/Gradient.cc\
#	$(SRCDIR)/ManipFields.cc\

CFLAGS   := $(CFLAGS)

PSELIBS := Dataflow/Network Dataflow/Ports Dataflow/XMLUtil \
	Core/Datatypes Dataflow/Widgets Core/Persistent Core/Exceptions \
	Core/Thread Core/Containers Core/GuiInterface Core/Geom \
	Core/Datatypes Core/Geometry Core/TkExtensions \
	Core/Math Core/Util
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(FLEX_LIBS) -lm $(XML_LIBRARY) $(THREAD_LIBS)

include $(SRCTOP)/scripts/smallso_epilogue.mk

