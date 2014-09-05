# The contents of this file are subject to the University of Utah Public
# License (the "License"); you may not use this file except in compliance
# with the License.
# 
# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
# License for the specific language governing rights and limitations under
# the License.
# 
# The Original Source Code is SCIRun, released March 12, 2001.
# 
# The Original Source Code was developed by the University of Utah.
# Portions created by UNIVERSITY are Copyright (C) 2001, 1994
# University of Utah. All Rights Reserved.
# 
#   File   : sub.mk
#   Author : David Weinstein
#   Date   : Mon Jul  1 17:54:36 MDT 2002

# Makefile fragment for this subdirectory

SRCDIR := Packages/rtrt/StandAlone/utils

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := Core/Datatypes Core/Containers Core/Persistent Core/Exceptions Core/Thread Core/Geometry Core/Util Packages/rtrt/Core
endif
LIBS := $(TCL_LIBRARY) $(TK_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(GL_LIBRARY) $(X_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY)

PROGRAM := $(SRCDIR)/SphereImagesToEnvmap
SRCS := $(SRCDIR)/SphereImagesToEnvmap.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/test-ppm
SRCS := $(SRCDIR)/test-ppm.cc
include $(SCIRUN_SCRIPTS)/program.mk

PSELIBS := Core/Exceptions Core/Thread
LIBS := $(TEEM_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY)

PROGRAM := $(SRCDIR)/vq
SRCS := $(SRCDIR)/vq.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/pca-reconstruct
SRCS := $(SRCDIR)/pca-reconstruct.cc
# Does need teem
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/dilate
SRCS := $(SRCDIR)/dilate.cc
# Does need teem
include $(SCIRUN_SCRIPTS)/program.mk

ifeq ($(HAVE_LAPACK),yes)
PROGRAM := $(SRCDIR)/pca-image
SRCS := $(SRCDIR)/pca-image.cc
# Does need teem
LIBS += $(LAPACK_LIBRARY) $(F_LIBRARY)
include $(SCIRUN_SCRIPTS)/program.mk
endif

