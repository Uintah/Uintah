//static char *id="@(#) $Id$";

/*
 * SimpVolVis.cc
 *
 * Peter-Pike Sloan
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

#include <PSECommon/Modules/Salmon/Tex.h>
// #include <PSECore/Modules/Salmon/NormQuant.h>

#include <PSECore/Widgets/PointWidget.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace PSECore::Widgets;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::Math;

class SimpVolVis : public Module {
  ScalarFieldIPort *inscalarfield;

  ColorMapIPort* incolormap;
  
  GeometryOPort* ogeom;
   
  int init;
  CrowdMonitor widget_lock;
  PointWidget *widget;

  int mode;
  
  int geom_id;
  int widgetMoved;
  
  int field_id; // id for the scalar field...
  int cmap_id;  // id associated with color map...
  
  GeomTexVolRender      *rvol;  // this guy does all the work..
  
  GeomTrianglesP        *triangles;
    
  Point Smin,Smax;
  Vector ddv;

  TCLint avail_tex;
  int num_slices;
public:
  SimpVolVis( const clString& id);

  virtual ~SimpVolVis();
  virtual void widget_moved(int last);    
  virtual void execute();
  void tcl_command( TCLArgs&, void* );
};



Module* make_SimpVolVis( const clString& id) {
  return new SimpVolVis(id);
}

static clString module_name("SimpVolVis");
static clString widget_name("VolVisLocatorWidget");

SimpVolVis::SimpVolVis(const clString& id)
  : Module("SimpVolVis", id, Filter), widget_lock("SimpVolVis widget lock"),
    rvol(0), mode(1),num_slices(64),triangles(0),
    avail_tex("avail_tex", id, this)
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
  geom_id=0;

  init=0;
  widgetMoved=1;
}

SimpVolVis::~SimpVolVis()
{

}

void SimpVolVis::tcl_command( TCLArgs& args, void* userdata)
{
  if (args[1] == "Set") {
    if (args[2] == "Mode") {
      args[3].get_int(mode);
      if (rvol) {
	rvol->SetMode(mode);
	rvol->SetDoOther(!mode);
      }
    } else if (args[2] == "NumSlices") {
      int ns;
      args[3].get_int(ns);
      if (rvol)
	rvol->SetNumSlices(ns);
    } else if (args[2] == "SliceTransp") {
      double st;
      args[3].get_double(st);
      if (rvol)
	rvol->SetAlpha(st);
    }
  } else if (args[1] == "MoveWidget") {
      if (!widget) return;
      Point w(widget->ReferencePoint());
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
      widget->SetPosition(w);
      widget_moved(1);
  } else if (args[1] == "Clear") {
    if (rvol) {
      rvol->Clear();
      rvol->map2d = (unsigned char *)1; // set it to something...
    }
  } else {
    Module::tcl_command(args, userdata);
  }
}

void SimpVolVis::execute(void)
{
  ScalarFieldHandle sfield;
  int executemode;
  const clString base("vol");
  const clString modes("mode");

  if (!inscalarfield->get(sfield)) {
    return;
  }
  else if (!sfield.get_rep()) {
    return;
  }
  
  if (!sfield->getRGBase())
    return;

  ScalarFieldRGuchar *rgchar = sfield->getRGBase()->getRGUchar();

  if (!rgchar) {
    cerr << "Not a char field!\n";
    return;
  }

  int texture=avail_tex.get()*1024*1024;
  int texneed=rgchar->nx*rgchar->ny*rgchar->nz;
  int dx=1;
  int dy=1;
  int dz=1;
  while(texneed>texture){
    int nnx=rgchar->nx/dx;
    int nny=rgchar->ny/dy;
    int nnz=rgchar->nz/dz;
    if(nnx > nny && nny > nnz){
      dx*=2;
    } else if(nny>nnz){
      dy*=2;
    } else {
      dz*=2;
    }
    texneed=rgchar->nx/dx*rgchar->ny/dy*rgchar->nz/dz;
  }

  if(dx != 1 || dy != 1 || dz != 1){
    int nnx=rgchar->nx/dx;
    int nny=rgchar->ny/dy;
    int nnz=rgchar->nz/dz;
    cerr << "was: " << rgchar->nx << "x" << rgchar->ny << "x" << rgchar->nz << ", now: " << nnx << "x" << nny << "x" << nnz << '\n';
    ScalarFieldRGuchar* r=new ScalarFieldRGuchar();
    r->resize(nnx, nny, nnz);
    Point min, max;
    rgchar->get_bounds(min, max);
    r->set_bounds(min, max);
    for(int x=0;x<nnx;x++){
      for(int y=0;y<nny;y++){
	for(int z=0;z<nnz;z++){
	  r->grid(x,y,z)=rgchar->grid(x*dx,y*dy,z*dz);
	}
      }
    }
    rgchar=r;
  }

  int use_tcl_stuff=1;

  if (!get_tcl_intvar(base,modes,executemode)) {
      cerr << "Can't find tcl variables in SimpVolVis!\n";
      use_tcl_stuff=0;
      executemode=0;
  }
  if (!init) {
    init=1;
    widget=scinew PointWidget(this, &widget_lock, 0.2);
    GeomObj *w=widget->GetWidget();
    ogeom->addObj(w, widget_name, &widget_lock);
    widget->Connect(ogeom);
    
// DAVE: HACK!
//    sfield->get_bounds(Smin, Smax);

    Point tmp1, tmp2;
    sfield->get_bounds(tmp1, tmp2);
    Smin=Point(tmp1.z(), tmp1.y(), tmp1.x());
    Smax=Point(tmp2.z(), tmp2.y(), tmp2.x());
    Vector dv(Smax-Smin);
    ddv.x(dv.x()/(rgchar->nz - 1));
    ddv.y(dv.y()/(rgchar->ny - 1));
    ddv.z(dv.z()/(rgchar->nx - 1));

    cerr << "Smin="<<Smin<<"  Smax="<<Smax<<"\n";
    widget->SetPosition(Interpolate(Smin,Smax,0.5));
    
    widget->SetScale(sfield->longest_dimension()/80.0);
    field_id = sfield->generation;

    triangles = scinew GeomTrianglesP();
    

    {
      Point w(widget->ReferencePoint());
    
      Vector vX(dv.x(), 0, 0);
      Vector vY(0, dv.y(), 0);
      Vector vZ(0, 0, dv.z());
    
      Point cornerX(w.x(), Smin.y(), Smin.z());
      Point cornerY(Smin.x(), w.y(), Smin.z());
      Point cornerZ(Smin.x(), Smin.y(), w.z());
      if (triangles->size()) {
	triangles->reserve_clear(6);
      }
      triangles->add(cornerZ,cornerZ+vX,cornerZ+vX+vY);
      triangles->add(cornerZ,cornerZ+vX+vY,cornerZ+vY);
    
      triangles->add(cornerY,cornerY+vX,cornerY+vX+vZ);
      triangles->add(cornerY,cornerY+vX+vZ,cornerY+vZ);
    
      triangles->add(cornerX,cornerX+vZ,cornerX+vZ+vY);
      triangles->add(cornerX,cornerX+vZ+vY,cornerX+vY);
    }
    
    //ogeom->addObj(triangles,"Cutting Planes TransParent");
  }

  int nx,ny,nz;

//  if (!get_tcl_intvar(base,clString("nx"),nx) ||
//      !get_tcl_intvar(base,clString("ny"),ny) ||
//      !get_tcl_intvar(base,clString("nz"),nz)) {
//      cerr << "NxNyNz don't exist...\n";
      use_tcl_stuff=0;
//    }	

  if (!rvol) {
    Point pmin,pmax;
	
// DAVE: HACK!
//    rgchar->get_bounds(pmin,pmax);
 
    Point tmp1, tmp2;
    rgchar->get_bounds(tmp1, tmp2);
    pmin=Point(tmp1.z(), tmp1.y(), tmp1.x());
    pmax=Point(tmp2.z(), tmp2.y(), tmp2.x());

   rvol = scinew GeomTexVolRender(pmin,pmax);
    rvol->SetVol((unsigned char*)&rgchar->grid(0,0,0),rgchar->nz,rgchar->ny,rgchar->nx);

    if (use_tcl_stuff)
	rvol->SubVol(nx,ny,nz);

    rvol->SetOther(triangles);
    rvol->SetNumSlices(256);
    rvol->SetAlpha(0.075);
    ogeom->addObj(rvol,"VolRender");
  } else {
    rvol->map2d = (unsigned char *)1; // force flush...
    if (executemode == 0) {
      rvol->SetVol((unsigned char*)&rgchar->grid(0,0,0),
		   rgchar->nz,rgchar->ny,rgchar->nx);

// DAVE: HACK!
//      sfield->get_bounds(Smin, Smax);
 
      Point tmp1, tmp2;
      sfield->get_bounds(tmp1, tmp2);
      Smin=Point(tmp1.z(), tmp1.y(), tmp1.x());
      Smax=Point(tmp2.z(), tmp2.y(), tmp2.x());

      Vector dv(Smax-Smin);
      ddv.x(dv.x()/(rgchar->nz - 1));
      ddv.y(dv.y()/(rgchar->ny - 1));
      ddv.z(dv.z()/(rgchar->nx - 1));
      
      Point w(widget->ReferencePoint());
      
      Vector vX(dv.x(), 0, 0);
      Vector vY(0, dv.y(), 0);
      Vector vZ(0, 0, dv.z());
      
      Point cornerX(w.x(), Smin.y(), Smin.z());
      Point cornerY(Smin.x(), w.y(), Smin.z());
      Point cornerZ(Smin.x(), Smin.y(), w.z());
      if (triangles->size()) {
	triangles->reserve_clear(6);
      }
      triangles->add(cornerZ,cornerZ+vX,cornerZ+vX+vY);
      triangles->add(cornerZ,cornerZ+vX+vY,cornerZ+vY);
      
      triangles->add(cornerY,cornerY+vX,cornerY+vX+vZ);
      triangles->add(cornerY,cornerY+vX+vZ,cornerY+vZ);
      
      triangles->add(cornerX,cornerX+vZ,cornerX+vZ+vY);
      triangles->add(cornerX,cornerX+vZ+vY,cornerX+vY);
    
    }
    if (use_tcl_stuff)
	rvol->SubVol(nx,ny,nz);
  }


  ColorMapHandle cmaph;
  if (incolormap->get(cmaph)) {
      // through it in the volume...

      rvol->map1d = cmaph->raw1d;
  }
  
  
  ogeom->flushViews();
}



void SimpVolVis::widget_moved(int /*last*/)
{
  if( !mode )
    {
      Vector dv(Smax-Smin);
      
      Point w(widget->ReferencePoint());
      
      Vector vX(dv.x(), 0, 0);
      Vector vY(0, dv.y(), 0);
      Vector vZ(0, 0, dv.z());
      
      Point cornerX(w.x(), Smin.y(), Smin.z());
      Point cornerY(Smin.x(), w.y(), Smin.z());
      Point cornerZ(Smin.x(), Smin.y(), w.z());
      
      // now you just have to put these triangles into the
      // textured triangle node thing...
      
      if (triangles->size()) {
	triangles->reserve_clear(6);
      }
      triangles->add(cornerZ,cornerZ+vX,cornerZ+vX+vY);
      triangles->add(cornerZ,cornerZ+vX+vY,cornerZ+vY);
      
      triangles->add(cornerY,cornerY+vX,cornerY+vX+vZ);
      triangles->add(cornerY,cornerY+vX+vZ,cornerY+vZ);
      
      triangles->add(cornerX,cornerX+vZ,cornerX+vZ+vY);
      triangles->add(cornerX,cornerX+vZ+vY,cornerX+vY);
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  1999/08/29 00:46:48  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/25 03:48:10  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:59  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:10  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:53  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:16  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:58:01  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:34  dav
// Import sources
//
//
