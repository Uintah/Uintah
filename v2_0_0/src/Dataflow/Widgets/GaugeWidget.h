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
 *  GaugeWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Gauge_Widget_h
#define SCI_project_Gauge_Widget_h 1

#include <Dataflow/Widgets/BaseWidget.h>

namespace SCIRun {

class GaugeWidget : public BaseWidget {
public:
  GaugeWidget( Module* module, CrowdMonitor* lock, double widget_scale,
	       bool is_slideable = false);
  GaugeWidget( const GaugeWidget& );
  virtual ~GaugeWidget();

  virtual void redraw();
  virtual void geom_pick(GeomPickHandle, ViewWindow*, int, const BState& bs);
  virtual void geom_moved(GeomPickHandle, int, double, const Vector&, int,
			  const BState&, const Vector &pick_offset);

  virtual void MoveDelta( const Vector& delta );
  virtual Point ReferencePoint() const;

  void SetRatio( const double ratio );
  double GetRatio() const;

  void SetEndpoints( const Point& end1, const Point& end2 );
  void GetEndpoints( Point& end1, Point& end2 ) const;

  const Vector& GetAxis();

  // Variable indexs
  enum { PointLVar, PointRVar, DistVar, SDistVar, RatioVar};

  // Material indexs
  enum { PointMatl, ShaftMatl, ResizeMatl, SliderMatl };

protected:
  virtual string GetMaterialName( const Index mindex ) const;   
   
private:
  Vector oldaxis;
  bool is_slideable_;

  Point pick_pointlvar_;
  Point pick_pointrvar_;
  double pick_distvar_;
  double pick_sdistvar_;
  double pick_ratiovar_;
};


} // End namespace SCIRun

#endif
