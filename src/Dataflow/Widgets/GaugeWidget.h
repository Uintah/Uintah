
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

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off warnings about partially overridden virtual functions
#pragma set woff 1682
#endif

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

} // End namespace SCIRun

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1682
#endif


#endif
