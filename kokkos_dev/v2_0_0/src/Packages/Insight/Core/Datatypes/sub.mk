#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.


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

INCLUDES += $(INSIGHT_INCLUDE)
SRCDIR   := Packages/Insight/Core/Datatypes

SRCS     += $(SRCDIR)/ITKDatatype.cc \
	$(SRCDIR)/ITKLatVolField.h \
	$(SRCDIR)/ITKImageField.h \
	$(SRCDIR)/cd_templates.cc \
	$(SRCDIR)/cd_templates_fields_0.cc \
	$(SRCDIR)/cd_templates_fields_1.cc \

#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Persistent Core/Exceptions Core/Containers \
	Core/Thread Core/Geometry Core/GuiInterface \
	Core/Math Core/Geom Core/Util Core/Datatypes 

LIBS := $(GL_LIBRARY) $(M_LIBRARY) $(INSIGHT_LIBRARY)


include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

