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


#  Original author: James Bigler

# Makefile fragment for tclIndex

# This makefile is designed to only create the tclIncex for the files
# which are specified in the make file with these two variables.

# $(SRCDIR) - The directory which contains the source files
# $(SRCS)   - The list of tcl files which should be included in the tclIndex
#           - each files is prepended with $(SRCDIR)/

# This is the target

TCLINDEX := $(SRCDIR)/tclIndex

# This is the mangles files which we need to pass to createTclIndex
# and in turn to auto_mkindex.  We need to make the filenames local,
# so we use patsubst to fix that for us.

$(TCLINDEX)_FILES := $(patsubst $(SRCDIR)/%,%, $(SRCS))

# This is the path to the tcl directory

$(TCLINDEX)_DIR := $(SRCTOP)/$(SRCDIR)

# Set up the dependencies and make rule.  We need to pass in two
# arguments.  The first one is the directory where our tcl files are
# the second is the list of files we wish to include (which is mangles
# based on the path to tclIndex).  The list of files is blocked by
# single quotes so that createTclIndex can call it as a single
# argument.

$(TCLINDEX): $(SRCS) $(SRCDIR)/sub.mk
	$(OBJTOP)/createTclIndex $($@_DIR) '$($@_FILES)'

# Add this to the list of targets

ALLTARGETS := $(ALLTARGETS) $(TCLINDEX)
ALLTCLINDEXES := $(ALLTCLINDEXES) $(TCLINDEX)

# Add this to the clean build
CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

# Try to prevent user error
SRCS := INVALID_SRCS.cc

