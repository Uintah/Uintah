//static char *id="@(#) $Id$";

/*
 *  ImageViewer.cc:  
 *
 *  Written by:
 *   ??
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRGshort.h>
#include <SCICore/Geom/tGrid.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class ImageViewer : public Module {
  ScalarFieldIPort *inscalarfield;
  GeometryOPort* ogeom;

  int grid_id;

  ScalarFieldRGshort *ingrid; // this only works on regular grids for chaining

  int u_num, v_num;
  Point corner;
  Vector u, v;
  ScalarField* sfield;
  TexGeomGrid* grid;
public:
  ImageViewer(const clString& id);
  virtual ~ImageViewer();
  virtual void execute();
};

Module* make_ImageViewer(const clString& id) {
  return new ImageViewer(id);
}

//static clString module_name("ImageViewer");

ImageViewer::ImageViewer(const clString& id)
: Module("ImageViewer", id, Filter)
{
  // Create the input ports
  // Need a scalar field and a colormap
  inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					  ScalarFieldIPort::Atomic);
  add_iport( inscalarfield);
  
  // Create the output port
  ogeom = scinew GeometryOPort(this, "Geometry", 
			       GeometryIPort::Atomic);
  add_oport(ogeom);
}

ImageViewer::~ImageViewer()
{
}

void ImageViewer::execute()
{
  int old_grid_id = grid_id;

  // get the scalar field and colormap...if you can
  ScalarFieldHandle sfieldh;
  if (!inscalarfield->get( sfieldh ))
    return;
  sfield=sfieldh.get_rep();

  if (!sfield->getRGBase())
    return;

  ingrid = sfield->getRGBase()->getRGShort();

  if (!ingrid)
    return;

  if (ingrid->grid.dim3() != 1) {
    error( "This module for 2d images only..");
    return;
  }
  
  u_num = ingrid->grid.dim1();
  v_num = ingrid->grid.dim2();
  
  corner = Point(0,0,0);
  u = Vector(u_num,0,0);
  v = Vector(0,v_num,0);

  cerr << u_num << " " << v_num << "\n";
  
  grid = scinew TexGeomGrid(v_num, u_num, corner, v, u,1);

  grid->set((unsigned short *) &ingrid->grid(0,0,0),4); // value doesn't matter...
  
  // delete the old grid/cutting plane
  if (old_grid_id != 0)
    ogeom->delObj( old_grid_id );
  
  grid_id = ogeom->addObj(grid, "Image Viewer");
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.7  1999/10/07 02:07:06  sparker
// use standard iostreams and complex type
//
// Revision 1.6  1999/08/29 00:46:46  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/25 03:48:08  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:58  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:07  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:50  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:14  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:58  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:34  dav
// Import sources
//
//
