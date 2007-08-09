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

SRCDIR := Framework/StandAlone

########################################################################
#
# SCIJump Stuff:
#


########################################################################
#
# Framework executable
#

#ifeq ($(LARGESOS),yes)
#  PSELIBS := Core/CCA
#else
  PSELIBS := \
             Framework/sidl \
             Core/Containers Core/Exceptions Core/Thread Core/Util
#  ifeq ($(HAVE_GLOBUS),yes)
#	  PSELIBS += Core/globus_threads
#  endif
#endif

LIBS := $(GLOBUS_LIBRARY) $(BABEL_LIBRARY)

#ifeq ($(HAVE_MPI),yes)
#  LIBS += $(MPI_LIBRARY)
#endif

ifeq ($(HAVE_WX),yes)
  LIBS += $(WX_LIBRARY)
endif

PROGRAM := $(FWK_EXE)
bundle_prologue: $(SRCTOP)/scripts/scijump/Info.plist.in
	mkdir -p $(FWK_APP)/Contents
	echo -n "APPL????" >$(FWK_APP)/Contents/PkgInfo
	mkdir -p $(FWK_APP)/Contents/MacOS
	mkdir -p $(FWK_APP)/Contents/Resources
	sed -e "s%KEYS%$(WX_DIR)/lib:$(OBJTOP_ABS)/lib%" \
          $(SRCTOP)/scripts/scijump/Info.plist.in > $(FWK_APP)/Contents/Info.plist
	cp -f $(SRCTOP)/scripts/scijump/scijump $(OBJTOP)/scijump

SRCS := $(SRCDIR)/main.cc
ifeq ($(OS_NAME),Darwin)
  PROGRAM_LDFLAGS := -bind_at_load
  #ALLTARGETS := bundle_prologue $(ALLTARGETS) bundle_epilogue
  #ALLTARGETS := bundle_prologue $(ALLTARGETS)
  ALLTARGETS := $(ALLTARGETS)
endif

include $(SCIRUN_SCRIPTS)/program.mk
