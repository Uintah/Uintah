
/*
 *  SquareWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Square_Widget_h
#define SCI_project_Square_Widget_h 1

#include <Widgets/BaseWidget.h>

enum { SquareW_PointUL, SquareW_PointUR, SquareW_PointDR, SquareW_PointDL,
       SquareW_Dist, SquareW_Hypo };

class SquareWidget : public BaseWidget {
public:
   SquareWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   SquareWidget( const SquareWidget& );
   ~SquareWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);
};


#endif
