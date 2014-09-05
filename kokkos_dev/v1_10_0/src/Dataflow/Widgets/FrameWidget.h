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
