
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

#include <Widgets/BaseWidget.h>

namespace PSECommon {
namespace Widgets {

class GaugeWidget : public BaseWidget {
public:
   GaugeWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   GaugeWidget( const GaugeWidget& );
   virtual ~GaugeWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetRatio( const Real ratio );
   Real GetRatio() const;

   void SetEndpoints( const Point& end1, const Point& end2 );
   void GetEndpoints( Point& end1, Point& end2 ) const;

   const Vector& GetAxis();

   // Variable indexs
   enum { PointLVar, PointRVar, DistVar, SDistVar, RatioVar};

   // Material indexs
   enum { PointMatl, ShaftMatl, ResizeMatl, SliderMatl };

protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
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
