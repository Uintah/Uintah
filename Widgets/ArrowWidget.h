
/*
 *  ArrowWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Arrow_Widget_h
#define SCI_project_Arrow_Widget_h 1

#include <Widgets/BaseWidget.h>

enum { ArrowW_Point };

class ArrowWidget : public BaseWidget {
public:
   ArrowWidget( Module* module );
   ArrowWidget( const ArrowWidget& );
   ~ArrowWidget();

   virtual void execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   inline void SetDirect( const Vector& v );
   inline const Vector& GetDirect() const;

private:
   Vector direction;
};


inline void
ArrowWidget::SetDirect( const Vector& v )
{
   direction = v.normal();
}


inline const Vector&
ArrowWidget::GetDirect() const
{
   return direction;
}


#endif
