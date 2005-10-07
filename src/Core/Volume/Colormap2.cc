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

#define COLORMAP2_VERSION 4

void
ColorMap2::io(Piostream &stream)
{
  const int version = stream.begin_class("ColorMap2", COLORMAP2_VERSION);

  if (version >= 2)
    PropertyManager::io(stream);

  bool faux = false;
  if (version <= 3)
    SCIRun::Pio(stream, faux);

  SCIRun::Pio(stream, widgets_);

  if (version <= 3)
    for (unsigned int w = 0; w < widgets_.size(); ++w)
      widgets_[w]->set_faux(faux);
      
  if (version >= 3)
    SCIRun::Pio(stream, selected_);

  if (version >= 4)
    SCIRun::Pio(stream, value_range_);

  stream.end_class();
}

ColorMap2::ColorMap2()
  : updating_(false),
    widgets_(),
    selected_(-1),
    value_range_(0.0, -1.0)
{}

ColorMap2::ColorMap2(const ColorMap2 &copy)
  : updating_(copy.updating_),
    widgets_(copy.widgets_),
    selected_(copy.selected_),
    value_range_(copy.value_range_)
{}

ColorMap2::ColorMap2(const vector<CM2WidgetHandle>& widgets,
                     bool updating,
                     bool selected,
                     pair<float, float> value_range)
  : updating_(updating),
    widgets_(widgets),
    selected_(selected),
    value_range_(value_range)
{}

ColorMap2::~ColorMap2()
{}

} // End namespace SCIRun
