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

SRCDIR := Packages/rtrt/StandAlone/tex-utils

PSELIBS := Core/Exceptions Core/Thread
LIBS := \
	$(TEEM_LIBRARY) \
	$(FASTM_LIBRARY) \
	$(M_LIBRARY) \
	$(THREAD_LIBRARY) \
	$(PERFEX_LIBRARY)

# genpttex
ifeq ($(LARGESOS),yes)
  PSELIBS += Packages/rtrt
else
  PSELIBS += \
	Packages/rtrt/Core \
	Packages/rtrt/visinfo \
	Core/Persistent \
	Core/Geometry
endif
PSELIBS += Packages/rtrt/Core/PathTracer
PROGRAM := $(SRCDIR)/genpttex
SRCS := $(SRCDIR)/genpttex.cc
include $(SCIRUN_SCRIPTS)/program.mk

# rm-black
PROGRAM := $(SRCDIR)/rm-black
SRCS := $(SRCDIR)/rm-black.cc
include $(SCIRUN_SCRIPTS)/program.mk

# pnn-vq
PROGRAM := $(SRCDIR)/pnn-vq
SRCS := $(SRCDIR)/pnn-vq.cc
include $(SCIRUN_SCRIPTS)/program.mk

# vq-error
PROGRAM := $(SRCDIR)/vq-error
SRCS := $(SRCDIR)/vq-error.cc
include $(SCIRUN_SCRIPTS)/program.mk

# recon-pca
PROGRAM := $(SRCDIR)/recon-pca
SRCS := $(SRCDIR)/recon-pca.cc
include $(SCIRUN_SCRIPTS)/program.mk

# pca-error
PROGRAM := $(SRCDIR)/pca-error
SRCS := $(SRCDIR)/pca-error.cc
include $(SCIRUN_SCRIPTS)/program.mk

# batch-pca
ifeq ($(HAVE_LAPACK),yes)
LIBS += $(LAPACK_LIBRARY) $(F_LIBRARY)
PROGRAM := $(SRCDIR)/batch-pca
SRCS := $(SRCDIR)/batch-pca.cc
include $(SCIRUN_SCRIPTS)/program.mk
endif

# serial_batch-pca
ifeq ($(HAVE_LAPACK),yes)
LIBS += $(LAPACK_LIBRARY) $(F_LIBRARY)
PROGRAM := $(SRCDIR)/serial_batch-pca
SRCS := $(SRCDIR)/serial_batch-pca.cc
include $(SCIRUN_SCRIPTS)/program.mk
endif
