
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

namespace PSECommon {
namespace Widgets {

enum LightType { DirectionalLight, PointLight, SpotLight, AreaLight, NumLightTypes };

class FrameWidget;

class LightWidget : public BaseWidget {
public:
   LightWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   LightWidget( const LightWidget& );
   ~LightWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

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

protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   LightType ltype;

   FrameWidget* arealight;

   Vector oldaxis;
};

} // End namespace Widgets
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:56:07  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:24  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif
