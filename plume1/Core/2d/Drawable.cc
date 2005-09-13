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
 *  Drawable.cc: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Malloc/Allocator.h>
#include <Core/2d/Drawable.h>

namespace SCIRun {

PersistentTypeID Drawable::type_id("Drawable", "Datatype", 0);

Drawable::Drawable( const string &name) 
  : name_(name), enabled_(true), parent_(0), ogl_(0)
{
  lock_ = scinew Mutex( name.c_str() );
}


Drawable::~Drawable()
{
}

void Drawable::reset_bbox()
{
  // Nothing to do, by default.
}

void Drawable::io(Piostream&)
{
  // Nothing for now...
}

void Pio( Piostream & stream, Drawable *& obj )
{
  Persistent* tmp=obj;
  stream.io(tmp, Drawable::type_id);
  if(stream.reading())
    obj=(Drawable*)tmp;
}

} // End namespace SCIRun
