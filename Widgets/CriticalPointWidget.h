
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

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, int);

   virtual void NextMode();

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetCriticalType( const CriticalType crit );
   Index GetCriticalType() const;

   void SetPosition( const Point& );
   Point GetPosition() const;
   
   void SetDirection( const Vector& v );
   const Vector& GetDirection() const;

   // Variable indexs
   enum { PointVar };

private:
   CriticalType crittype;
   Vector direction;
};


#endif
