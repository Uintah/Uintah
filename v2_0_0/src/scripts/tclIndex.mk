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
#  Portions created by UNIVERSITY are Copyright (C) 2003, 1994 
#  University of Utah. All Rights Reserved.
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

# Add this to the clean build
CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

# Try to prevent user error
SRCS := INVALID_SRCS.cc

