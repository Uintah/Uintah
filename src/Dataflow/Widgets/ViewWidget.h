
/*
 *  ViewWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_View_Widget_h
#define SCI_project_View_Widget_h 1

#include <PSECore/Widgets/BaseWidget.h>
#include <SCICore/Geom/View.h>

namespace PSECore {
namespace Widgets {

using SCICore::GeomSpace::View;

class ViewWidget : public BaseWidget {
public:
   ViewWidget( Module* module, CrowdMonitor* lock, double widget_scale,
	       const Real AspectRatio=1.3333);
   ViewWidget( const ViewWidget& );
   virtual ~ViewWidget();

   virtual void redraw();
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   View GetView();
   Vector GetUpVector();
   Real GetFOV() const;

   void SetView( const View& view );

   Real GetAspectRatio() const;
   void SetAspectRatio( const Real aspect );
   
   const Vector& GetEyeAxis();
   const Vector& GetUpAxis();
   Point GetFrontUL();
   Point GetFrontUR();
   Point GetFrontDR();
   Point GetFrontDL();
   Point GetBackUL();
   Point GetBackUR();
   Point GetBackDR();
   Point GetBackDL();

   // Variable indexs
   enum { EyeVar, ForeVar, LookAtVar, UpVar, UpDistVar, EyeDistVar, FOVVar };

   // Material indexs
   enum { EyesMatl, ResizeMatl, ShaftMatl, FrustrumMatl };
   
protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   Real ratio;
   Vector oldaxis1;
   Vector oldaxis2;
};

} // End namespace Widgets
} // End namespace PSECore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:33  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:10  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:26  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif
