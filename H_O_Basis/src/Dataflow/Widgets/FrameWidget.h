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
 *  FrameWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_ScaledFrame_Widget_h
#define SCI_project_ScaledFrame_Widget_h 1

#include <Dataflow/Widgets/BaseWidget.h>


namespace SCIRun {

class FrameWidget : public BaseWidget {
public:
  FrameWidget( Module* module, CrowdMonitor* lock, double widget_scale,
		     bool is_slideable = false);
  FrameWidget( const FrameWidget& );
  virtual ~FrameWidget();

  virtual void redraw();
  virtual void geom_moved(GeomPickHandle, int, double, const Vector&, int,
			  const BState&, const Vector &pick_offset);
  virtual void geom_pick(GeomPickHandle, ViewWindow*, int, const BState& bs);

  virtual void MoveDelta( const Vector& delta );
  virtual Point ReferencePoint() const;

  void SetPosition( const Point& center, const Point& R, const Point& D );
  void GetPosition( Point& center, Point& R, Point& D );

  void SetPosition( const Point& center, const Vector& normal,
		    const double size1, const double size2 );
  void GetPosition( Point& center, Vector& normal,
		    double& size1, double& size2 );

  void SetRatioR( const double ratio );
  double GetRatioR() const;
  void SetRatioD( const double ratio );
  double GetRatioD() const;

  void SetSize( const double sizeR, const double sizeD );
  void GetSize( double& sizeR, double& sizeD ) const;
   
  const Vector& GetRightAxis();
  const Vector& GetDownAxis();

  // Variable indexs
  enum { CenterVar, PointRVar, PointDVar, DistRVar, DistDVar, HypoVar,
	 SDistRVar, RatioRVar, SDistDVar, RatioDVar, NumVars };

  // Material indexs
  enum { PointMatl, EdgeMatl, ResizeMatl, SliderMatl, NumMatls };

  virtual string GetStateString();
  virtual int SetStateString(const string &str);

protected:
  virtual string GetMaterialName( const Index mindex ) const;   
   
private:
  bool is_slideable_;

  Vector oldrightaxis, olddownaxis;
  Point pick_centervar_;
  Point pick_pointrvar_;
  Point pick_pointdvar_;

  Vector rot_start_ray_;
};


} // End namespace SCIRun

#endif
