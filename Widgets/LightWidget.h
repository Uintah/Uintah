
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

#include <Widgets/BaseWidget.h>


enum LightType { DirectionalLight, PointLight, SpotLight, AreaLight, NumLightTypes };

class FrameWidget;

class LightWidget : public BaseWidget {
public:
   LightWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   LightWidget( const LightWidget& );
   ~LightWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, int);

   virtual void NextMode();

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetLightType( const LightType lighttype );
   LightType GetLightType() const;
   
   const Vector& GetAxis();

   // Variable indexs
   enum { SourceVar, DirectVar, ConeVar, DistVar, RadiusVar, RatioVar };

   // Variable Indexs
   enum { SourceMatl, ArrowMatl, PointMatl, ConeMatl };

private:
   LightType ltype;

   FrameWidget* arealight;

   Vector oldaxis;
};


#endif
