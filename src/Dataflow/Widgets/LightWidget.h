
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

#include <PSECore/Widgets/BaseWidget.h>

namespace PSECore {
namespace Widgets {

enum LightType { DirectionalLight, PointLight, SpotLight, AreaLight, NumLightTypes };

class FrameWidget;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

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
} // End namespace PSECore

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1682
#endif

//
// $Log$
// Revision 1.3  1999/10/07 02:07:25  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/08/17 06:38:30  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:07  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:24  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif
