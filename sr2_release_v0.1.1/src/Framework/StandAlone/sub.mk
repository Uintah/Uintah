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

ifeq ($(LARGESOS),yes)
  PSELIBS := Core/CCA
else
  PSELIBS := \
             Framework Core/CCA/PIDL Core/CCA/spec Core/CCA/SSIDL \
             Core/Containers Core/Exceptions Core/Thread Core/Util
  ifeq ($(HAVE_GLOBUS),yes)
	  PSELIBS += Core/globus_threads
  endif
endif

LIBS := $(GLOBUS_LIBRARY)

ifeq ($(HAVE_BABEL), yes)
  LIBS += $(SIDL_LIBRARY)
endif

ifeq ($(HAVE_MPI),yes)
  LIBS += $(MPI_LIBRARY)
endif

ifeq ($(HAVE_WX),yes)
  LIBS += $(WX_LIBRARY)
endif

$(FWK_EXE).app: $(FWK_EXE) $(SRCTOP)/scripts/scijump/Info.plist.in $(SRCTOP)/scripts/scijump/wxmac.icns
	mkdir -p $(FWK_EXE).app/Contents
	mkdir -p $(FWK_EXE).app/Contents/MacOS
	mkdir -p $(FWK_EXE).app/Contents/Resources
	cp -f $(SRCTOP)/scripts/scijump/Info.plist.in $(FWK_EXE).app/Contents/Info.plist
	echo -n "APPL????" >$(FWK_EXE).app/Contents/PkgInfo
	ln -f $(FWK_EXE) $(FWK_EXE).app/Contents/MacOS/$(FWK_EXE)
	cp -f $(SRCTOP)/scripts/scijump/wxmac.icns $(FWK_EXE).app/Contents/Resources/wxmac.icns
#sed -e "s/IDENTIFIER/`echo $(SRCTOP) | sed -e 's,\.\./,,g' | sed -e 's,/,.,g'`/" \ -e "s/EXECUTABLE/splitter/" \ -e "s/VERSION/$(FWK_VERSION)/" \ $(SRCTOP)/scripts/scijump/Info.plist.in >$(FWK_EXE).app/Contents/Info.plist


PROGRAM := $(FWK_EXE)
SRCS := $(SRCDIR)/main.cc
ifeq ($(OS_NAME),Darwin)
  ALLTARGETS := $(ALLTARGETS) $(FWK_EXE).app
endif
include $(SCIRUN_SCRIPTS)/program.mk

########################################################################
#
# ploader
#

# build the SCIJump CCA Component Loader here
ifeq ($(LARGESOS),yes)
  PSELIBS := Core/CCA/Component
else
  PSELIBS := \
             Framework Core/CCA/PIDL Core/CCA/DT Core/CCA/spec Core/CCA/SSIDL \
             Core/Exceptions Core/Thread
  ifeq ($(HAVE_GLOBUS),yes)
	PSELIBS += Core/globus_threads
  endif
endif

ifeq ($(HAVE_BABEL), yes)
  LIBS += $(SIDL_LIBRARY)
endif

ifeq ($(HAVE_MPI),yes)
  LIBS := $(MPI_LIBRARY)
endif

PROGRAM := ploader
SRCS := $(SRCDIR)/ploader.cc
include $(SCIRUN_SCRIPTS)/program.mk

########################################################################
