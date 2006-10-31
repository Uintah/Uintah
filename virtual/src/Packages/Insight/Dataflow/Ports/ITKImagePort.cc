/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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


/*
 *  ITKImagePort.cc
 *
 *  Written by:
 *   Darby J Brown
 *   Department of Computer Science
 *   University of Utah
 *   January 2003
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Packages/Insight/Dataflow/Ports/ITKImagePort.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Insight/Dataflow/Ports/share.h>
namespace Insight {

using namespace SCIRun;

extern "C" {
  SCISHARE IPort* make_ITKImageIPort(Module* module, const string& name) {
  return scinew SimpleIPort<SCIRun::ITKImageHandle>(module,name);
}
  SCISHARE OPort* make_ITKImageOPort(Module* module, const string& name) {
  return scinew SimpleOPort<SCIRun::ITKImageHandle>(module,name);
}
}
} // End namespace Insight

namespace SCIRun {
template<> string SimpleIPort<SCIRun::ITKImageHandle>::port_type_("ITKImage");
template<> string SimpleIPort<SCIRun::ITKImageHandle>::port_color_("pink");
} // End namespace SCIRun

