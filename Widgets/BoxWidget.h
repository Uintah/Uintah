
/*
 *  BoxWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Box_Widget_h
#define SCI_project_Box_Widget_h 1

#include <Widgets/BaseWidget.h>

enum { BoxW_PointIUL, BoxW_PointIUR, BoxW_PointIDR, BoxW_PointIDL,
       BoxW_PointOUL, BoxW_PointOUR, BoxW_PointODR, BoxW_PointODL,
       BoxW_Dist, BoxW_Hypo, BoxW_Diag };

class BoxWidget : public BaseWidget {
public:
   BoxWidget( Module* module, double widget_scale );
   BoxWidget( const BoxWidget& );
   ~BoxWidget();

   virtual void execute();
   virtual void geom_moved(int, double, const Vector&, void*);
};


#endif
