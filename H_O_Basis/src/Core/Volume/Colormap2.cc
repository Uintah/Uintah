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
//    File   : ColorMap2.cc
//    Author : Milan Ikits
//    Date   : Mon Jul  5 18:33:29 2004

#include <Core/Util/NotFinished.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Volume/Colormap2.h>
#include <Core/Volume/CM2Widget.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

static Persistent* maker()
{
  return scinew ColorMap2;
}

PersistentTypeID ColorMap2::type_id("ColorMap2", "Datatype", maker);

#define COLORMAP2_VERSION 1

void
ColorMap2::io(Piostream &stream)
{
  stream.begin_class("ColorMap2", COLORMAP2_VERSION);
  
  SCIRun::Pio(stream, faux_);
  SCIRun::Pio(stream, widgets_);

  stream.end_class();
}

ColorMap2::ColorMap2()
  : updating_(false)
{}

ColorMap2::ColorMap2(const vector<CM2WidgetHandle>& widgets,
		     bool updating, bool faux)
  : updating_(updating),
    faux_(faux)
{
  for(unsigned int i=0; i<widgets.size(); i++)
    widgets_.push_back(widgets[i]->clone());
}

ColorMap2::~ColorMap2()
{
}

} // End namespace SCIRun
