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
 *  PointWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Point_Widget_h
#define SCI_project_Point_Widget_h 1

#include <Dataflow/Widgets/BaseWidget.h>
#include <Core/Geom/Material.h>

namespace SCIRun {

class PointWidget : public BaseWidget {
public:
  PointWidget( Module* module, CrowdMonitor* lock, double widget_scale );
  PointWidget( const PointWidget& );
  virtual ~PointWidget();

  virtual void redraw();
  virtual void geom_pick(GeomPickHandle, ViewWindow*, int, const BState& bs);
  virtual void geom_moved(GeomPickHandle, int, double, const Vector&,
			  int, const BState&, const Vector &pick_offset);

  virtual void MoveDelta( const Vector& delta );
  virtual Point ReferencePoint() const;

  void SetPosition( const Point& );
  Point GetPosition() const;

  virtual void widget_tcl( GuiArgs& );

  // Variable indexs
  enum { PointVar };

  // Material indexs
  enum { PointMatl };
   
protected:
  virtual string GetMaterialName( const Index mindex ) const;   

  Point pick_position_;
};


} // End namespace SCIRun

#endif
