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
 *  LightWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Light_Widget_h
#define SCI_project_Light_Widget_h 1

#include <Dataflow/Widgets/BaseWidget.h>


namespace SCIRun {

enum LightType {
  DirectionalLight=0, PointLight=1, SpotLight=2, AreaLight=3, NumLightTypes=4
};

class FrameWidget;

class LightWidget : public BaseWidget {
public:
   LightWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   LightWidget( Module* module, CrowdMonitor* lock, 
		double widget_scale, Point source,
		Point direct, Point cone, double rad, double rat );
   LightWidget( const LightWidget& );
   
   ~LightWidget();
   void init( Module* module);
   virtual void redraw();
   virtual void geom_moved(GeomPickHandle, int, double, const Vector&, int,
			   const BState&, const Vector &pick_offset);

   virtual void NextMode();

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetLightType( const LightType lighttype );
   LightType GetLightType() const;
   
  const Vector& GetAxis();
  Point GetSource() const;
  Point GetPointAt() const;
  Point GetCone() const;
  void SetPointAt( const Point& p );
  void SetCone( const Point& p );
  double GetRadius() const;
  void SetRadius( double r );
  double GetRatio() const;
  void SetRatio( double r );

   // Variable indexs
   enum { SourceVar, DirectVar, ConeVar, DistVar, RadiusVar, RatioVar };

   // Variable Indexs
   enum { SourceMatl, ArrowMatl, PointMatl, ConeMatl };

protected:
   virtual string GetMaterialName( const Index mindex ) const;   
   
private:
   LightType ltype;

   FrameWidget* arealight;

   Vector oldaxis;
};


} // End namespace SCIRun

#endif
