#
#  For more information, please see: http://software.sci.utah.edu
#
#  The MIT License
#
#  Copyright (c) 2007 Scientific Computing and Imaging Institute,
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

SRCDIR := Components/Babel/tutorial/randomgens

include $(SCIRUN_SCRIPTS)/babel_prologue.mk

#
# For languages other than C++, include a babel language makefile fragement here.
# Ex. $(SCIRUN_SCRIPTS)/babel_component_f77.mk for a Fortran 77 server
#

SERVER_SIDL := $(SRCDIR)/randomgens.sidl \
                Components/Babel/tutorial/ports/randomgen.sidl

include $(SCIRUN_SCRIPTS)/babel_server.mk

#
# For languages other than C++, include a babel language makefile fragement here.
# Ex. $(SCIRUN_SCRIPTS)/babel_component_f77.mk for a Fortran 77 client
#

#CLIENT_SIDL := 

#include $(SCIRUN_SCRIPTS)/babel_client.mk

#
# Put component-specific SCIRun libraries (PSELIBS), third-party libraries
# (LIBS) and append to includes (INCLUDES) here.
#

PSELIBS := 
LIBS :=

include $(SCIRUN_SCRIPTS)/babel_epilogue.mk
