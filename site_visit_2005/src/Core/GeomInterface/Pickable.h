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
 *  Pickable.h: ???
 *
 *  Written by:
 *   Dav de St. Germain...
 *   Department of Computer Science
 *   University of Utah
 *   Mar 1999
 *
 *  Copyright (C) 1999 University of Utah
 */

#ifndef SCI_Geom_Pickable_h
#define SCI_Geom_Pickable_h 1

#include <Core/Geom/GeomPick.h>

namespace SCIRun {

class Vector;
class Point;
class ViewWindow;
class GeomPick;

struct BState {
   unsigned int control:1;
   unsigned int alt:1;
   unsigned int shift:1;
   unsigned int btn:2;
};


class WidgetPickable {

public:
  virtual ~WidgetPickable();

  virtual void geom_pick(GeomPickHandle, ViewWindow*, int widget_data, 
			 const BState& bs);
  virtual void geom_release(GeomPickHandle, int, const BState& bs);
  virtual void geom_moved(GeomPickHandle, int, double, const Vector&, int,
			  const BState&, const Vector &pick_offset);
};


class ModulePickable {

public:
  virtual ~ModulePickable();

  virtual void geom_pick(GeomPickHandle, void*, GeomHandle);
  virtual void geom_release(GeomPickHandle, void*, GeomHandle);
  virtual void geom_moved(GeomPickHandle, int, double, const Vector&,
			  void*, GeomHandle);
};

} // End namespace SCIRun

#endif /* SCI_Geom_Pickable_h */
