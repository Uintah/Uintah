
/*
 *  InterfaceWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Interface_Widget_h
#define SCI_project_Interface_Widget_h 1

#include "BaseWidget.h"

const Index NumInterfacePoints = 4;


class InterfaceWidget : public BaseWidget {
public:
   InterfaceWidget();
   InterfaceWidget( const InterfaceWidget& );
   ~InterfaceWidget();

private:
};


#endif
