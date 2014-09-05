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
//    File   : Colormap2Port.cc
//    Author : Milan Ikits
//    Date   : Mon Jul  5 18:46:29 2004

#include <Dataflow/Ports/Colormap2Port.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

extern "C" {

SCIRun::IPort* make_ColorMap2IPort(SCIRun::Module* module,
                                                const std::string& name) {
  return scinew SCIRun::SimpleIPort<ColorMap2Handle>(module,name);
}
  
SCIRun::OPort* make_ColorMap2OPort(SCIRun::Module* module,
                                                const std::string& name) {
  return scinew SCIRun::SimpleOPort<ColorMap2Handle>(module,name);
}

}

template<> std::string SCIRun::SimpleIPort<ColorMap2Handle>::port_type_("ColorMap2");
template<> std::string SCIRun::SimpleIPort<ColorMap2Handle>::port_color_("darkseagreen");

} // End namespace SCIRun

