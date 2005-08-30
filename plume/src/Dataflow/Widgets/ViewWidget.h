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
 *  ViewWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_View_Widget_h
#define SCI_project_View_Widget_h 1

#include <Dataflow/Widgets/BaseWidget.h>
#include <Core/Geom/View.h>


namespace SCIRun {


class ViewWidget : public BaseWidget {
public:
  ViewWidget( Module* module, CrowdMonitor* lock, double widget_scale,
	      double AspectRatio = 1.3333);
  ViewWidget( const ViewWidget& );
  virtual ~ViewWidget();

  virtual void redraw();
  virtual void geom_moved(GeomPickHandle, int, double, const Vector&, int,
			  const BState&, const Vector &pick_offset);

  virtual void MoveDelta( const Vector& delta );
  virtual Point ReferencePoint() const;

  View GetView();
  Vector GetUpVector();
  double GetFOV() const;

  void SetView( const View& view );

  double GetAspectRatio() const;
  void SetAspectRatio( const double aspect );
   
  const Vector& GetEyeAxis();
  const Vector& GetUpAxis();
  Point GetFrontUL();
  Point GetFrontUR();
  Point GetFrontDR();
  Point GetFrontDL();
  Point GetBackUL();
  Point GetBackUR();
  Point GetBackDR();
  Point GetBackDL();

  // Variable indexs
  enum { EyeVar, ForeVar, LookAtVar, UpVar, UpDistVar, EyeDistVar, FOVVar };

  // Material indexs
  enum { EyesMatl, ResizeMatl, ShaftMatl, FrustrumMatl };
   
protected:
  virtual string GetMaterialName( const Index mindex ) const;   
   
private:
  double ratio;
  Vector oldaxis1;
  Vector oldaxis2;
};


} // End namespace SCIRun

#endif
