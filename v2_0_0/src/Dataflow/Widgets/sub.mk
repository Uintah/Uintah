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

SRCDIR   := Dataflow/Widgets

SRCS +=	$(SRCDIR)/ArrowWidget.cc           \
	$(SRCDIR)/BaseWidget.cc		   \
	$(SRCDIR)/BoxWidget.cc		   \
	$(SRCDIR)/CriticalPointWidget.cc   \
	$(SRCDIR)/CrosshairWidget.cc	   \
	$(SRCDIR)/FrameWidget.cc	   \
	$(SRCDIR)/GaugeWidget.cc	   \
	$(SRCDIR)/LightWidget.cc	   \
	$(SRCDIR)/PathWidget.cc		   \
	$(SRCDIR)/PointWidget.cc	   \
	$(SRCDIR)/RingWidget.cc		   \
	$(SRCDIR)/ViewWidget.cc

PSELIBS := Core/Datatypes Dataflow/Constraints Dataflow/Ports \
	Core/Exceptions Core/Geom Core/Thread Core/GuiInterface \
	Core/Containers Core/Geometry Core/GeomInterface Dataflow/Network
LIBS := $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


ifeq ($(LARGESOS),no)
SCIRUN_MODULES := $(SCIRUN_MODULES) $(LIBNAME)
endif
