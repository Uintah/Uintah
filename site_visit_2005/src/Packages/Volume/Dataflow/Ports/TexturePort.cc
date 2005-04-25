//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : TexturePort.cc
//    Author : Milan Ikits
//    Date   : Thu Jul 15 15:01:21 2004

#include <Packages/Volume/Dataflow/Ports/TexturePort.h>
#include <Dataflow/share/share.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;

namespace Volume {

extern "C" {
  
PSECORESHARE IPort* make_TextureIPort(Module* module,
                                      const string& name) {
  return scinew SimpleIPort<TextureHandle>(module,name);
}
  
PSECORESHARE OPort* make_TextureOPort(Module* module,
                                      const string& name) {
  return scinew SimpleOPort<TextureHandle>(module,name);
}

}

template<> string SimpleIPort<TextureHandle>::port_type_("Texture");
template<> string SimpleIPort<TextureHandle>::port_color_("wheat");

} // namespace Volume
