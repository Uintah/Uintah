/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  RingWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Ring_Widget_h
#define SCI_project_Ring_Widget_h 1

#include <Dataflow/Widgets/BaseWidget.h>


namespace SCIRun {

class RingWidget : public BaseWidget {
public:
  RingWidget( Module* module, CrowdMonitor* lock, double widget_scale );
  RingWidget( const RingWidget& );
  virtual ~RingWidget();

  virtual void redraw();
  virtual void geom_pick(GeomPickHandle, ViewWindow*, int, const BState& bs);
  virtual void geom_moved(GeomPickHandle, int, double, const Vector&, int,
			  const BState&, const Vector &pick_offset);

  virtual void MoveDelta( const Vector& delta );
  virtual Point ReferencePoint() const;

  void SetPosition( const Point& center, const Vector& normal, const double radius );
  void GetPosition( Point& center, Vector& normal, double& radius ) const;
   
  void SetRatio( const double ratio );
  double GetRatio() const;

  void GetPlane( Vector& v1, Vector& v2);

  void SetRadius( const double radius );
  double GetRadius() const;
   
  const Vector& GetRightAxis();
  const Vector& GetDownAxis();

  // Variable indexs
  enum { CenterVar, PointRVar, PointDVar, DistVar, HypoVar, Sqrt2Var,
	 SliderVar, AngleVar };

  // Materials indexs
  enum { PointMatl, RingMatl, SliderMatl, ResizeMatl, HalfResizeMatl };
   
protected:
  virtual string GetMaterialName( const Index mindex ) const;   
   
private:
  Vector oldrightaxis, olddownaxis;

  Point pick_centervar_;
  Point pick_pointrvar_;
  Point pick_pointdvar_;
  Point pick_slidervar_;
};


} // End namespace SCIRun

#endif
