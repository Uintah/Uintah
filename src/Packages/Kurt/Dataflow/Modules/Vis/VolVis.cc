
/*
 * VolVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarFieldRG.h>

#include <Core/Geom/GeomTriangles.h>

#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>

#include <Dataflow/Widgets/PointWidget.h>
#include <iostream>
#include "VolVis.h"
#include <Packages/Kurt/Geom/MultiBrick.h>
#include <Packages/Kurt/Geom/VolumeUtils.h>



namespace Kurt {

using namespace SCIRun;
using std::cerr;

static clString widget_name("VolVisLocatorWidget");
static clString res_name("Resolution Widget");
			 
extern "C" Module* make_VolVis( const clString& id) {
  return new VolVis(id);
}

VolVis::VolVis(const clString& id)
  : Module("VolVis", id, Filter), widget_lock("VolVis widget lock"),
    mode(0), draw_mode("draw_mode", id, this), debug("debug", id, this),
    alpha("alpha", id, this),
    influence("influence", id, this),
    num_slices("num_slices", id, this) ,
    avail_tex("avail_tex", id, this),
    max_brick_dim("max_brick_dim", id, this), level("level", id, this),
    brick(0), res_lock("VolVis resolution lock"), widget(0), res(0),
    widget_id(-1), res_id(-1)
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
      if( brick ){
	brick->SetMode( mode );
      }	
      cerr<< "res_id = "<<res_id<<endl;
      cerr<< "widget_id = "<<widget_id<<endl;
      if( mode == 3){
	if(res_id >= 0){
	  ogeom->delObj( res_id, 0 );
	  res_id = -1;
	}
	if(widget_id >= 0){
	  ogeom->delObj( widget_id, 0 );
	  widget_id = -1;
	}	  
      } else if(!mode ) {
	if( widget){
	  GeomObj *w=widget->GetWidget();
	  widget_id = ogeom->addObj(w, widget_name, &widget_lock);
	}
	if(res_id >= 0){
	  ogeom->delObj( res_id, 0 );
	  res_id = -1;
	}
      } else {
	if( res ){
	  GeomObj *r=res->GetWidget();
	  res_id = ogeom->addObj(r, res_name, &res_lock);
	}
	if( widget_id >= 0 ){
	  ogeom->delObj( widget_id, 0 );
	  widget_id = -1;
	}
      }
      
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
    } else if (args[2] == "Influence"){
      double inf;
      args[3].get_double(inf);
      influence.set(inf);
      cerr<< "Influence = " << inf<<endl;
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
      cerr<< "MoveWidgit " << w << endl;
  } else if (args[1] == "Clear") {
      cerr << "Clear "<< endl;
  } else {
    Module::tcl_command(args, userdata);
  }
}

void VolVis::widget_moved(int obj)
{
  //  cerr<<"the widget id is "<<obj<<endl;
  //  cerr<<"is brick set? "<< (( brick == 0)? "NO":"YES")<<endl;
  //  cerr<<"mode is "<<mode<<endl;
  if( !mode && brick )
    {
      brick->SetPlaneIntersection(widget->ReferencePoint());
      cerr<<"moving widget to "<<widget->ReferencePoint()<<endl;
    } else {
      brick->SetResPosition(res->ReferencePoint());
      cerr<<"moving widget to "<<res->ReferencePoint()<<endl;
    }
  
}


void VolVis::SwapXZ( ScalarFieldHandle sfh )
{
  ScalarFieldRGuchar *ifu, *ofu;
  ifu = sfh->getRGBase()->getRGUchar();
  int nx=ifu->nx;
  int ny=ifu->ny;
  int nz=ifu->nz;
  Point min;
  Point max;
  sfh->get_bounds(min, max);

  ofu = scinew ScalarFieldRGuchar();
  ofu->resize(nz,ny,nx);
  ofu->set_bounds(min, max);
  for (int i=0, ii=0; i<nx; i++, ii++)
    for (int j=0, jj=0; j<ny; j++, jj++)
      for (int k=0, kk=0; k<nz; k++, kk++)
	ofu->grid(k,j,i)=ifu->grid(ii,jj,kk);

  sfh = ScalarFieldHandle( ofu );

}
  
void VolVis::execute(void)
{
  static ScalarFieldHandle sfield;
  static ScalarFieldHandle field = 0;
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
    if( field.get_rep() != sfield.get_rep() ){
      SwapXZ( sfield );
      field = sfield;
      rgchar = sfield->getRGBase()->getRGUchar();
    }
    int nx, ny, nz;
    int padx = 0, pady = 0, padz = 0;
    Point pmin,pmax;
    rgchar->get_bounds(pmin, pmax);

    if(!widget){
      widget=scinew PointWidget(this, &widget_lock, 0.2);
      GeomObj *w=widget->GetWidget();
      if( !draw_mode.get() )
	widget_id = ogeom->addObj(w, widget_name, &widget_lock);
      widget->Connect(ogeom);
    
      // DAVE: HACK!
      //    sfield->get_bounds(Smin, Smax);

      Smin=Point(pmin.z(), pmin.y(), pmin.x());
      Smax=Point(pmax.z(), pmax.y(), pmax.x());
      //      cerr << "Smin="<<Smin<<"  Smax="<<Smax<<"\n";
      widget->SetPosition(Interpolate(Smin,Smax,0.5));
      Vector dv(Smax-Smin);
      ddv.x(dv.x()/(rgchar->nz - 1));
      ddv.y(dv.y()/(rgchar->ny - 1));
      ddv.z(dv.z()/(rgchar->nx - 1));
      widget->SetScale(rgchar->longest_dimension()/80.0);
    }

    if( !res ){
      res = scinew PointWidget(this, &res_lock, 0.2);
      GeomObj *r = res->GetWidget();
      if(draw_mode.get() == 1 || draw_mode.get() ==2)
	res_id = ogeom->addObj(r,res_name, &res_lock);
      res->Connect(ogeom);
      res->SetPosition(Point(0,0,0));
      res->SetScale(rgchar->longest_dimension()/50.0);
    }
    
    
      //ogeom->addObj(triangles,"Cutting Planes TransParent");
    if( !brick ){
      brick = new MultiBrick( 0x12345676, num_slices.get(), alpha.get(),
			      max_brick_dim.get(), pmin, pmax,
			      draw_mode.get(), debug.get(),
			      rgchar->nz, rgchar->ny, rgchar->nx,
			      rgchar, (unsigned char*)cmap->raw1d);
      brick->SetDrawLevel(level.get());
    
      if( widget )
	brick->SetPlaneIntersection(widget->ReferencePoint());
      if( res )
	brick->SetResPosition(res->ReferencePoint());
      int l = brick->getMaxLevel();
      int dim = brick->getMaxSize();
      TCL::execute( id + " SetDims " + to_string( dim ));
      TCL::execute( id + " SetLevels " + to_string( l ));
      ogeom->addObj( brick, "TexBrick" ); 

    } else {
      brick->Reload();
      brick->SetMaxBrickSize( max_brick_dim.get(), max_brick_dim.get(),
			       max_brick_dim.get());
      brick->SetColorMap((unsigned char*)cmap->raw1d);
      brick->SetDebug( debug.get());
      brick->SetAlpha( alpha.get());
      brick->SetNSlices( num_slices.get());
      brick->SetInfluence ( influence.get());
      brick->SetVol( rgchar );
      //      brick->SetMode( mode );
      brick->SetDrawLevel(level.get());
      //brick->SetPlaneIntersection(widget->ReferencePoint());
      int l = brick->getMaxLevel();
      int dim = brick->getMaxSize();
      TCL::execute( id + " SetDims " + to_string( dim ));
      TCL::execute( id + " SetLevels " + to_string( l ));

      Vector dv(Smax-Smin);
      ddv.x(dv.x()/(rgchar->nz - 1));
      ddv.y(dv.y()/(rgchar->ny - 1));
      ddv.z(dv.z()/(rgchar->nx - 1));

      ogeom->flushViews();
    }
  }
} // End execute(void);

} // End namespace Kurt

