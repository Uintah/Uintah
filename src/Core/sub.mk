#
#  Copyright (c) 1997-2012 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the \"Software\"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
# 
# 
# 
# 
# 
# Makefile fragment for this subdirectory


include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Core

SUBDIRS := \
	$(SRCDIR)/Basis \
	$(SRCDIR)/Containers \
	$(SRCDIR)/DataArchive \
	$(SRCDIR)/Datatypes \
	$(SRCDIR)/Disclosure \
	$(SRCDIR)/Exceptions \
	$(SRCDIR)/Geometry \
	$(SRCDIR)/GeometryPiece \
	$(SRCDIR)/Grid \
	$(SRCDIR)/Labels \
	$(SRCDIR)/IO \
	$(SRCDIR)/Malloc \
	$(SRCDIR)/Math \
	$(SRCDIR)/OS \
	$(SRCDIR)/Parallel \
	$(SRCDIR)/Persistent \
	$(SRCDIR)/ProblemSpec \
	$(SRCDIR)/Thread \
	$(SRCDIR)/Tracker \
	$(SRCDIR)/Util \
#	$(SRCDIR)/ICom \
#	$(SRCDIR)/2d \
#	$(SRCDIR)/Util/Comm \
#	$(SRCDIR)/Parts \
#	$(SRCDIR)/PartsGui \
#[INSERT NEW CATEGORY DIR HERE]



include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS := $(THREAD_LIBRARY) \
	$(Z_LIBRARY) $(M_LIBRARY) 

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk

