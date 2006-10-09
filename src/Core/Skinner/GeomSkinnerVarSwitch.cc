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



/*
 *  SkinnerVarSwitch.cc:  Turn Geometry on and off
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1995
 *
 *  Copyright (C) 1995 SCI Group
 */
#include <Core/Skinner/GeomSkinnerVarSwitch.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomSkinnerVarSwitch()
{
  Skinner::Var<bool> foo(0,"");
    return new GeomSkinnerVarSwitch(0,foo);
}

PersistentTypeID GeomSkinnerVarSwitch::type_id("GeomSkinnerVarSwitch", "GeomObj", make_GeomSkinnerVarSwitch);

GeomSkinnerVarSwitch::GeomSkinnerVarSwitch(GeomHandle obj, 
                                           const Skinner::Var<bool> &var)
: GeomContainer(obj), 
  state_(var)
{
}

GeomSkinnerVarSwitch::GeomSkinnerVarSwitch(const GeomSkinnerVarSwitch& copy)
: GeomContainer(copy), state_(copy.state_)
{
}

GeomObj*
GeomSkinnerVarSwitch::clone()
{
    return new GeomSkinnerVarSwitch(*this);
}

void
GeomSkinnerVarSwitch::set_state(bool state)
{
   state_ = state;
}

bool
GeomSkinnerVarSwitch::get_state()
{
   return state_;
}

void
GeomSkinnerVarSwitch::get_bounds(BBox& bbox)
{
   if (state_ && child_.get_rep()) child_->get_bounds(bbox);
}


void
GeomSkinnerVarSwitch::fbpick_draw(DrawInfoOpenGL* di, Material* matl, double time)
{
  if (state_ && child_.get_rep())
  {
    child_->fbpick_draw(di, matl, time);
  }
}

void
GeomSkinnerVarSwitch::draw(DrawInfoOpenGL* di, Material* matl, double time)
{
  if (state_ && child_.get_rep())
  {
    child_->draw(di, matl, time);
  }
}


void GeomSkinnerVarSwitch::io(Piostream& stream)
{
    stream.begin_class("GeomSkinnerVarSwitch", 1);
    GeomContainer::io(stream);
    bool state = state_;
    Pio(stream, state);
    state_ = state;
    stream.end_class();
}

}
