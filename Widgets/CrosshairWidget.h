
/*
 *  CrosshairWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Crosshair_Widget_h
#define SCI_project_Crosshair_Widget_h 1

#include <Widgets/BaseWidget.h>


// Variable indexs
enum { CrosshairW_Center };
// Material indexs
enum { CrosshairW_CenterMatl, CrosshairW_AxesMatl, CrosshairW_HighMatl };


class CrosshairWidget : public BaseWidget {
public:
   CrosshairWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   CrosshairWidget( const CrosshairWidget& );
   ~CrosshairWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   // They should be orthogonal.
   inline void SetAxes( const Vector& v1, const Vector& v2, const Vector& v3 );

private:
   Vector axis1, axis2, axis3;
};


inline void
CrosshairWidget::SetAxes( const Vector& v1, const Vector& v2, const Vector& v3 )
{
   if ((v1.length2() > 1e-6)
       && (v2.length2() > 1e-6)
       && (v3.length2() > 1e-6)) {
      axis1 = v1.normal();
      axis2 = v2.normal();
      axis3 = v3.normal();
   }
}


#endif
