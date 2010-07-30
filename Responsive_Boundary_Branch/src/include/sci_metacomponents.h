/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

#ifndef include_sci_metacomponents_h
#define include_sci_metacomponents_h

// determine if meta-component model bridging should be enabled or not

#include <sci_defs/framework_defs.h>
#include <sci_defs/babel_defs.h>
#include <sci_defs/dataflow_defs.h>
#include <sci_defs/vtk_defs.h>
#include <sci_defs/tao_defs.h>
#include <sci_defs/ruby_defs.h>

// There are currently too many SCIRun Dataflow dependencies
// in the Bridge component model to build it without SCIRun dataflow.
#if defined (HAVE_RUBY) && defined (HAVE_BABEL) && defined (BUILD_DATAFLOW)
#  define BUILD_BRIDGE 1
#endif

#include <Framework/CCA/CCAComponentModel.h>
#include <Framework/Internal/InternalComponentModel.h>

#ifdef BUILD_DATAFLOW
#  include <Framework/Dataflow/SCIRunComponentModel.h>
#endif

#ifdef BUILD_BRIDGE
#  include <Framework/Bridge/BridgeComponentModel.h>
#endif
#if HAVE_BABEL
#  include <Framework/Babel/BabelComponentModel.h>
#endif
#if HAVE_VTK
#  include <Framework/Vtk/VtkComponentModel.h>
#endif
#if HAVE_TAO
#  include <Framework/Corba/CorbaComponentModel.h>
#  include <Framework/Tao/TaoComponentModel.h>
#endif

#endif
