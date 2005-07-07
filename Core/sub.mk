#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Core

SUBDIRS := \
	$(SRCDIR)/Algorithms \
	$(SRCDIR)/Bundle \
	$(SRCDIR)/Containers \
	$(SRCDIR)/Datatypes \
	$(SRCDIR)/Exceptions \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Comm \
	$(SRCDIR)/Geom \
	$(SRCDIR)/GeomInterface \
	$(SRCDIR)/Geometry \
	$(SRCDIR)/GuiInterface \
	$(SRCDIR)/ImportExport \
	$(SRCDIR)/Init \
	$(SRCDIR)/Malloc \
	$(SRCDIR)/Math \
	$(SRCDIR)/OS \
	$(SRCDIR)/Persistent \
	$(SRCDIR)/Process \
	$(SRCDIR)/Services \
	$(SRCDIR)/SystemCall \
	$(SRCDIR)/Thread \
	$(SRCDIR)/TkExtensions \
	$(SRCDIR)/Util \
	$(SRCDIR)/Volume \
	$(SRCDIR)/2d \
	$(SRCDIR)/ICom \
#	$(SRCDIR)/Util/Comm \
#	$(SRCDIR)/Parts \
#	$(SRCDIR)/PartsGui \
#[INSERT NEW CATEGORY DIR HERE]


ifeq ($(BUILD_SCIRUN2),yes)
SUBDIRS := \
	$(SUBDIRS) \
	$(SRCDIR)/CCA \
	$(SRCDIR)/Babel 
endif

ifeq ($(HAVE_GLOBUS),yes)
SUBDIRS+=$(SRCDIR)/globus_threads
endif

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS := $(PLPLOT_LIBRARY) $(BLT_LIBRARY) $(ITCL_LIBRARY) $(TCL_LIBRARY) \
	$(TK_LIBRARY) $(ITK_LIBRARY) $(GL_LIBRARY) $(THREAD_LIBRARY) \
	$(Z_LIBRARY) $(M_LIBRARY) 

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk

