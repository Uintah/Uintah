//static char *id="@(#) $Id$";


/**************************************
CLASS
   ClipField
   Takes a field as input and clips the field according to a bounding
   box provided by the UI.  The new field is sent to the oport.

GENERAL INFORMATION
   ClipField.cc

   Written by:
   David Weinstein (& Eric Kuehne)
   Department of Computer Science
   University of Utah
   February 1995 (& July 2000)

   Copyright (C) 2000 SCI Group
   

KEYWORDS
   clip, clipping, bounding_box

DESCRIPTION
   Takes a field as input and creates a new ClipFieldAlgo object based
   on the type of input field.  The ClipFieldAlgo::clip(field*, bbox)
   method clips returns a copy of the field clipped to the bbox.  The
   method handles both unstructured and structured fields, and both
   flat and indexed attribute types.   

WARNING
   none

****************************************/
#include <stdio.h>

#include "ClipFieldAlgo.h"

#include <SCICore/Datatypes/SField.h>
#include <SCICore/Datatypes/GenSField.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/FieldPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <PSECore/Widgets/ScaledBoxWidget.h>
#include <SCICore/Thread/CrowdMonitor.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Thread;
using namespace PSECore::Widgets;
using namespace SCICore::Geometry;

class ClipField : public Module {
public:
  // GROUP: Constructors:
  //////////
  // Construct ClipField with the id
  ClipField(const clString& id);

  // GROUP: Destructors:
  //////////
  // Destruction!
  virtual ~ClipField();
  
  // GROUP: Execution:
  //////////
  // Execute this module
  virtual void execute();
private:
  FieldIPort* ifield;
  FieldOPort* ofield;
  GeometryOPort* ogeom;
  TCLint x_min;
  TCLint x_max;
  TCLint y_min;
  TCLint y_max;
  TCLint z_min;
  TCLint z_max;
  TCLint sameInput;
  int first_time;
  int last_x_min;
  int last_y_min;
  int last_z_min;
  int last_x_max;
  int last_y_max;
  int last_z_max;

  int widget_id;
  CrowdMonitor widgetlock;
  ScaledBoxWidget* boxwidget;
};

extern "C" Module* make_ClipField(const clString& id) {
  return new ClipField(id);
}

static clString widget_name("IsoSurface widget");

ClipField::ClipField(const clString& id)
  : Module("ClipField", id, Filter), 
  x_min("x_min", id, this),y_min("y_min", id, this),z_min("z_min", id, this), 
  x_max("x_max", id, this),y_max("y_max", id, this),z_max("z_max", id, this),
  sameInput("sameInput", id, this), widgetlock("ClipField widget lock")
{
    ifield=new FieldIPort(this, "Field", FieldIPort::Atomic);
    add_iport(ifield);
    // Create the output port
    ofield=new FieldOPort(this, "Field", FieldIPort::Atomic);
    add_oport(ofield);
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    first_time=1;
    boxwidget = new ScaledBoxWidget(this, &widgetlock, 1.0);
}

ClipField::~ClipField()
{
}



void ClipField::execute()
{
  FieldHandle fh;
  ifield->get(fh);

  widget_id = ogeom->addObj(boxwidget->GetWidget(), widget_name, &widgetlock);
  boxwidget->Connect(ogeom);
  
  // Get Field
  // Get BBox for clipping region
  BBox bbox;
  // Decide if field is structured or unstructured
  ClipFieldAlgoBase* op = ClipFieldAlgoBase::make(fh.get_rep());
  Field* newfield = op->clip(fh.get_rep(), bbox);
  
}












  

     
  
  /*

  ScalarFieldHandle ifh;
    if(!ifield->get(ifh))
	return;
    ScalarFieldRGBase* isf=ifh->getRGBase();
    if(!isf){
	error("ClipField can't deal with unstructured grids!");
	return;
    }

    ScalarFieldRGdouble *ifd=isf->getRGDouble();
    ScalarFieldRGfloat *iff=isf->getRGFloat();
    ScalarFieldRGint *ifi=isf->getRGInt();
    ScalarFieldRGchar *ifc=isf->getRGChar();
    
    int mxx, mxy, mxz, mnx, mny, mnz;
    mxx=x_max.get()-1;
    mxy=y_max.get()-1;
    mxz=z_max.get()-1;
    mnx=x_min.get()-1;
    mny=y_min.get()-1;
    mnz=z_min.get()-1;
    if (!sameInput.get() || first_time || 
	mxx!=last_x_max || mxy!=last_y_max || mxz!=last_z_max ||
	mnx!=last_x_min || mny!=last_y_min || mnz != last_z_min) {
	first_time=0;
	if (ifd) {
	    ScalarFieldRGdouble *of;
  	    fldHandle = of = 0;
	    fldHandle = of = new ScalarFieldRGdouble;
	    of->resize(mxx-mnx+1, mxy-mny+1, mxz-mnz+1);
	    for (int i=0; i<=mxx-mnx; i++) {
		for (int j=0; j<=mxy-mny; j++) {
		    for (int k=0; k<=mxz-mnz; k++) {
			of->grid(i,j,k)=ifd->grid(i+mnx, j+mny, k+mnz);
		    }
		}
	    }
	    of->compute_minmax();
	    osf=of;
	} else if (iff) {
	    ScalarFieldRGfloat *of;
	    fldHandle = of = 0;
	    fldHandle = of = new ScalarFieldRGfloat;
	    of->resize(mxx-mnx+1, mxy-mny+1, mxz-mnz+1);
	    for (int i=0; i<=mxx-mnx; i++) {
	    for (int j=0; j<=mxy-mny; j++) {
		    for (int k=0; k<=mxz-mnz; k++) {
			of->grid(i,j,k)=iff->grid(i+mnx, j+mny, k+mnz);
		    }
		}
	    }
	    of->compute_minmax();
	    osf=of;
	} else if (ifi) {
	    ScalarFieldRGint *of;
	    fldHandle = of = 0;
	    fldHandle = of = new ScalarFieldRGint;
	    of->resize(mxx-mnx+1, mxy-mny+1, mxz-mnz+1);
	    for (int i=0; i<=mxx-mnx; i++) {
		for (int j=0; j<=mxy-mny; j++) {
		    for (int k=0; k<=mxz-mnz; k++) {
			of->grid(i,j,k)=ifi->grid(i+mnx, j+mny, k+mnz);
		    }
		}
	    }
	    of->compute_minmax();
	    osf=of;
	} else {
	    ScalarFieldRGchar *of;
	    fldHandle = of = 0;
	    fldHandle = of = new ScalarFieldRGchar;
	    of->resize(mxx-mnx+1, mxy-mny+1, mxz-mnz+1);
	    for (int i=0; i<=mxx-mnx; i++) {
		for (int j=0; j<=mxy-mny; j++) {
		    for (int k=0; k<=mxz-mnz; k++) {
			of->grid(i,j,k)=ifc->grid(i+mnx, j+mny, k+mnz);
		    }
		}
	    }
	    of->compute_minmax();
	    osf=of;
	}
	osf->set_bounds(Point(mnx-1, mny-1, mnz-1), 
			Point(mxx-1, mxy-1, mxz-1));
	last_x_max=mxx;
	last_y_max=mxy;
	last_z_max=mxz;
	last_x_min=mnx;
	last_y_min=mny;
	last_z_min=mnz;
    }
    ofield->send(osf);
}
    */

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6.2.1  2000/09/11 16:17:48  kuehne
// updates to field redesign
//
// Revision 1.6  2000/03/17 09:26:56  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.5  1999/08/25 03:47:45  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:42  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:38  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:26  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:40  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:48  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:32  dav
// Import sources
//
//
