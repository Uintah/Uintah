
/*
 *  GuageWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Guage_Widget_h
#define SCI_project_Guage_Widget_h 1

#include <Widgets/BaseWidget.h>


// Variable indexs
enum { GuageW_PointL, GuageW_PointR, GuageW_Dist, GuageW_Slider, GuageW_SDist, GuageW_Ratio};
// Material indexs
enum { GuageW_PointMatl, GuageW_EdgeMatl, GuageW_SliderMatl, GuageW_HighMatl };


class GuageWidget : public BaseWidget {
public:
   GuageWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   GuageWidget( const GuageWidget& );
   ~GuageWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   void SetRatio( const Real ratio );
   Real GetRatio() const;

   void SetEndpoints( const Point& end1, const Point& end2 );
   void GetEndpoints( Point& end1, Point& end2 ) const;

   const Vector& GetAxis();

private:
   Vector oldaxis;
};


#endif
