
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

enum { GuageW_PointL, GuageW_PointR, GuageW_Slider, GuageW_Dist };

class GuageWidget : public BaseWidget {
public:
   GuageWidget( Module* module );
   GuageWidget( const GuageWidget& );
   ~GuageWidget();

   virtual void execute();
   virtual void geom_moved(int, double, const Vector&, void*);
};


#endif
