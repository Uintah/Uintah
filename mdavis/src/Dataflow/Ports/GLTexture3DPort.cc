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



#include <Dataflow/Ports/GLTexture3DPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {



extern "C" {
IPort* make_GLTexture3DIPort(Module* module,
					 const string& name) {
  return scinew SimpleIPort<GLTexture3DHandle>(module,name);
}
OPort* make_GLTexture3DOPort(Module* module,
					 const string& name) {
  return scinew SimpleOPort<GLTexture3DHandle>(module,name);
}
}

template<> string SimpleIPort<GLTexture3DHandle>::port_type_("GLTexture3D");
template<> string SimpleIPort<GLTexture3DHandle>::port_color_("gray40");


} // End namespace SCIRun

