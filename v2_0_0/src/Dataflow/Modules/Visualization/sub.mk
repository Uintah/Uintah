#
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
#

# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/AddLight.cc\
	$(SRCDIR)/ChooseColorMap.cc\
	$(SRCDIR)/GLTextureBuilder.cc\
	$(SRCDIR)/GenStandardColorMaps.cc\
	$(SRCDIR)/GenTransferFunc.cc\
	$(SRCDIR)/Isosurface.cc\
	$(SRCDIR)/RescaleColorMap.cc\
	$(SRCDIR)/ShowColorMap.cc\
	$(SRCDIR)/ShowField.cc\
	$(SRCDIR)/ShowMatrix.cc\
	$(SRCDIR)/StreamLines.cc\
	$(SRCDIR)/TexCuttingPlanes.cc\
	$(SRCDIR)/TextureVolVis.cc\
#	$(SRCDIR)/TransformGeometry.cc\
[INSERT NEW CODE FILE HERE]




PSELIBS := Dataflow/Network Dataflow/Widgets Dataflow/Ports \
	Dataflow/Modules/Render Core/Datatypes Core/Containers \
	Core/Exceptions Core/Thread Core/GuiInterface Core/Geom \
	Core/Persistent Core/Geometry Core/Util \
	Core/TkExtensions Core/Algorithms/Visualization \
	Core/GLVolumeRenderer Core/GeomInterface

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
SCIRUN_MODULES := $(SCIRUN_MODULES) $(LIBNAME)
endif
