
/*
 *  CriticalPointWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_CriticalPoint_Widget_h
#define SCI_project_CriticalPoint_Widget_h 1

#include <Widgets/BaseWidget.h>


class CriticalPointWidget : public BaseWidget {
public:
   // Critical types
   enum CriticalType { Regular, AttractingNode, RepellingNode, Saddle,
		       AttractingFocus, RepellingFocus, SpiralSaddle,
		       NumCriticalTypes };

   CriticalPointWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   CriticalPointWidget( const CriticalPointWidget& );
   virtual ~CriticalPointWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void NextMode();

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetCriticalType( const CriticalType crit );
   Index GetCriticalType() const;

   void SetPosition( const Point& );
   Point GetPosition() const;
   
   void SetDirection( const Vector& v );
   const Vector& GetDirection() const;

   virtual void widget_tcl( TCLArgs& );

   // Variable indexs
   enum { PointVar };

   // Material indexs
   enum { PointMaterial, ShaftMaterial, HeadMaterial, CylinderMatl, TorusMatl, ConeMatl };

protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   CriticalType crittype;
   Vector direction;
};


#endif
