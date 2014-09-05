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

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Core/GLVolumeRenderer

SRCS +=	$(SRCDIR)/Brick.cc              \
	$(SRCDIR)/FullRes.cc		\
	$(SRCDIR)/FullResIterator.cc	\
	$(SRCDIR)/GLAttenuate.cc	\
	$(SRCDIR)/GLMIP.cc		\
	$(SRCDIR)/GLOverOp.cc		\
	$(SRCDIR)/GLPlanes.cc		\
	$(SRCDIR)/GLTexRenState.cc	\
	$(SRCDIR)/GLTexture3D.cc	\
	$(SRCDIR)/GLTextureIterator.cc	\
	$(SRCDIR)/GLVolRenState.cc	\
	$(SRCDIR)/GLVolumeRenderer.cc   \
	$(SRCDIR)/LOS.cc		\
	$(SRCDIR)/LOSIterator.cc	\
	$(SRCDIR)/ROI.cc		\
	$(SRCDIR)/ROIIterator.cc	\
	$(SRCDIR)/SliceTable.cc		\
	$(SRCDIR)/TexPlanes.cc		\
	$(SRCDIR)/VolumeUtils.cc

PSELIBS := Core/Persistent Core/Geometry Core/Exceptions \
	Core/Math Core/Containers Core/Thread Core/Datatypes \
	Core/GuiInterface Core/Geom Core/Util Core/Algorithms/GLVolumeRenderer

LIBS := $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

