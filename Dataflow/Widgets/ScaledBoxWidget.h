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
 *  ScaledBoxWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_project_ScaledBox_Widget_h
#define SCI_project_ScaledBox_Widget_h 1

#include <Dataflow/Widgets/BaseWidget.h>
#include <Core/Datatypes/Clipper.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

namespace SCIRun {

class ScaledBoxWidget : public BaseWidget {
public:
  ScaledBoxWidget( Module* module, CrowdMonitor* lock, double widget_scale,
		   bool is_aligned = false, bool is_slideable = true );
  ScaledBoxWidget( const ScaledBoxWidget& );
  virtual ~ScaledBoxWidget();

  virtual void redraw();
  virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);
  virtual void geom_pick(GeomPick*, ViewWindow*, int, const BState& bs);

  virtual void MoveDelta( const Vector& delta );
  virtual Point ReferencePoint() const;

  void SetPosition( const Point& center, const Point& R, const Point& D,
		    const Point& I );
  void GetPosition( Point& center, Point& R, Point& D, Point& I );

  void SetRatioR( const double ratio );
  double GetRatioR() const;
  void SetRatioD( const double ratio );
  double GetRatioD() const;
  void SetRatioI( const double ratio );
  double GetRatioI() const;

  const Vector& GetRightAxis();
  const Vector& GetDownAxis();
  const Vector& GetInAxis();

  bool IsAxisAligned() const { return is_aligned_; }
  bool IsSlideable() const { return is_slideable_; }

  // Variable indexs
  enum { CenterVar, PointRVar, PointDVar, PointIVar,
	 DistRVar, DistDVar, DistIVar, HypoRDVar, HypoDIVar, HypoIRVar,
	 SDistRVar, RatioRVar, SDistDVar, RatioDVar, SDistIVar, RatioIVar,
	 NumVars };

  // Material indexs
  enum { PointMatl, EdgeMatl, ResizeMatl, SliderMatl, NumMatls };

  ClipperHandle get_clipper();

protected:
  virtual string GetMaterialName( const Index mindex ) const;   
   
private:
  bool is_aligned_;
  bool is_slideable_;

  Vector oldrightaxis, olddownaxis, oldinaxis;
  Point rot_start_pt_;
  Point rot_start_d_;
  Point rot_start_r_;
  Point rot_start_i_;
  Vector rot_start_ray_norm_;
  Vector rot_curr_ray_;
};


} // End namespace SCIRun

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1682
#endif

#endif
