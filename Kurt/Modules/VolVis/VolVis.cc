
/*
 * VolVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ColorMap.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>

#include <SCICore/Geom/GeomTriangles.h>

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/CrowdMonitor.h>

#include <PSECore/Widgets/PointWidget.h>
#include <iostream>
#include "VolVis.h"
#include <Kurt/Geom/MultiBrick.h>
#include <Kurt/Geom/VolumeUtils.h>


unsigned char textureMem[32][32][32];

namespace Kurt {
namespace Modules {

using namespace Kurt::GeomSpace;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace PSECore::Widgets;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using std::cerr;


extern "C" Module* make_VolVis( const clString& id) {
  return new VolVis(id);
}


VolVis::VolVis(const clString& id)
  : Module("VolVis", id, Filter), widget_lock("VolVis widget lock"),
    mode(0), draw_mode("draw_mode", id, this), debug("debug", id, this),
    alpha("alpha", id, this), 
    num_slices("num_slices", id, this) ,
    avail_tex("avail_tex", id, this),
    max_brick_dim("max_brick_dim", id, this), level("level", id, this),
    brick(0)
{
  // Create the input ports
  inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					   ScalarFieldIPort::Atomic);

  add_iport(inscalarfield);
  incolormap=scinew  
    ColorMapIPort(this, "Color Map", ColorMapIPort::Atomic);
    
  add_iport(incolormap);
					
  // Create the output port
  ogeom = scinew GeometryOPort(this, "Geometry", 
			       GeometryIPort::Atomic);
  add_oport(ogeom);

  init=0;
  widgetMoved=1;
}

VolVis::~VolVis()
{

}

void VolVis::tcl_command( TCLArgs& args, void* userdata)
{
  if (args[1] == "Set") {
    if (args[2] == "Mode") {
      args[3].get_int(mode);
      cerr<< "Set Mode = "<< mode << endl;
    } else if (args[2] == "NumSlices") {
      int ns;
      args[3].get_int(ns);
      num_slices.set( ns );
      cerr<< "NumSlice = "<< ns << endl;
    } else if (args[2] == "SliceTransp") {
      double st;
      args[3].get_double(st);
      alpha.set(st);
      cerr<< "SliceTransp = " << st << endl;
    } else if (args[2] == "Dim"){
      int n;
      args[3].get_int( n );
      max_brick_dim.set(n);
    }
  } else if (args[1] == "MoveWidget") {
      if (!widget) return;
      Point w(0,0,0);
      if (args[2] == "xplus") {
	  w+=Vector(ddv.x(), 0, 0);
      } else if (args[2] == "xminus") {
	  w-=Vector(ddv.x(), 0, 0);
      } else if (args[2] == "yplus") {
	  w+=Vector(0, ddv.y(), 0);
      } else if (args[2] == "yminus") {
	  w-=Vector(0, ddv.y(), 0);
      } else if (args[2] == "zplus") {
	  w+=Vector(0, 0, ddv.z());
      } else {	// (args[3] == "zminus")
	  w-=Vector(0, 0, ddv.z());
      }
      //widget->SetPosition(w);
      //widget_moved(1);
      cerr<< "MoveWidgit " << w << endl;
  } else if (args[1] == "Clear") {
      cerr << "Clear "<< endl;
  } else {
    Module::tcl_command(args, userdata);
  }
}


void VolVis::execute(void)
{
  ScalarFieldHandle sfield;
  const clString base("draw");
  const clString modes("mode");

  if (!inscalarfield->get(sfield)) {
    return;
  }
  else if (!sfield.get_rep()) {
    return;
  }
  
  if (!sfield->getRGBase())
    return;

  ColorMapHandle cmap;
  if( !incolormap->get(cmap)){
    return;
  }
  
      
  ScalarFieldRGuchar *rgchar = sfield->getRGBase()->getRGUchar();

  if (!rgchar) {
    cerr << "Not a char field!\n";
    return;
  } else {
    int nx, ny, nz;
    int padx = 0, pady = 0, padz = 0;
    
    Point pmin,pmax;
    rgchar->get_bounds(pmin, pmax);

      cerr << "slices, alpha "<<num_slices.get()<<", "<<alpha.get()<<endl;
      cerr << "max dim, pmin, pmax, "<<max_brick_dim.get()<<", "<<pmin<<", "<< pmax<<endl;
      cerr << "field size "<<rgchar->nx <<", "<<rgchar->ny <<", "<<rgchar->nz <<endl;
    brick = new MultiBrick( 0x12345676, num_slices.get(), alpha.get(),
			  max_brick_dim.get(), pmin, pmax,
			  (draw_mode.get() == 2), debug.get(),
			  rgchar->nz, rgchar->ny, rgchar->nx,
			  rgchar, (unsigned char*)cmap->raw1d);
    brick->SetDrawLevel(level.get());
    
    int l = brick->getMaxLevel();
    int dim = brick->getMaxSize();
    TCL::execute( id + " SetDims " + to_string( dim ));
    TCL::execute( id + " SetLevels " + to_string( l ));


    ogeom->delAll();
    ogeom->addObj( brick, "TexBrick" );

    ogeom->flushViews();
  }
}

} // End namespace Modules
} // End namespace Uintah


