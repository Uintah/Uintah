/*
 *  FieldCage.cc:  IsoSurfaces a SFRG bitwise
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
//#include <Dataflow/Ports/VFieldPort.h>
#include <Core/Datatypes/SField.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/TclInterface/TCLvar.h>
#include <Core/Util/DebugStream.h>

namespace SCIRun {


class FieldCage : public Module {
  FieldIPort* insfield;
  //VectorFieldIPort* invfield;
  GeometryOPort* ogeom;
  MaterialHandle dk_red;
  MaterialHandle dk_green;
  MaterialHandle dk_blue;
  MaterialHandle lt_red;
  MaterialHandle lt_green;
  MaterialHandle lt_blue;
  MaterialHandle gray;
  TCLint numx;
  TCLint numy;
  TCLint numz;
  static DebugStream dbg;
public:
    FieldCage(const clString& id);
    virtual ~FieldCage();
    virtual void execute();
    MaterialHandle matl;
};

DebugStream FieldCage::dbg("FieldCage", true);

extern "C" Module* make_FieldCage(const clString& id) {
  return new FieldCage(id);
}

FieldCage::FieldCage(const clString& id)
: Module("FieldCage", id, Filter), numx("numx", id, this),
  numy("numy", id, this), numz("numz", id, this)
{
    // Create the input ports
    insfield=new FieldIPort(this, "Field", FieldIPort::Atomic);
    add_iport(insfield);
    //invfield=new VectorFieldIPort(this, "Vector Field", VectorFieldIPort::Atomic);
    //add_iport(invfield);
    // Create the output port
    ogeom=new GeometryOPort(this, "Geom", GeometryIPort::Atomic);
    add_oport(ogeom);
    matl=scinew Material(Color(0,0,0), Color(.8,.8,.8),
			 Color(.7,.7,.7), 50);
    dk_red = scinew Material(Color(0,0,0), Color(.3,0,0),
			     Color(.5,.5,.5), 20);
    dk_green = scinew Material(Color(0,0,0), Color(0,.3,0),
			       Color(.5,.5,.5), 20);
    dk_blue = scinew Material(Color(0,0,0), Color(0,0,.3),
			      Color(.5,.5,.5), 20);
    lt_red = scinew Material(Color(0,0,0), Color(.8,0,0),
			     Color(.5,.5,.5), 20);
    lt_green = scinew Material(Color(0,0,0), Color(0,.8,0),
			       Color(.5,.5,.5), 20);
    lt_blue = scinew Material(Color(0,0,0), Color(0,0,.8),
			      Color(.5,.5,.5), 20);
    gray = scinew Material(Color(0,0,0), Color(.4,.4,.4),
			   Color(.5,.5,.5), 20);
}

FieldCage::~FieldCage()
{
}

void FieldCage::execute()
{
  ogeom->delAll();
  
  FieldHandle sfh;
  BBox sbbox;
  if(insfield->get(sfh)){
    Field* field = sfh.get_rep();
    if(!field->get_geom()->get_bbox(sbbox)){
      error("FieldCage: Could not compute bounding box of input field.");
      return;
    }
  }
  else{
    return;
  }
  GeomGroup* all=new GeomGroup();
  GeomLines* xmin=new GeomLines();
  GeomLines* ymin=new GeomLines();
  GeomLines* zmin=new GeomLines();
  GeomLines* xmax=new GeomLines();
  GeomLines* ymax=new GeomLines();
  GeomLines* zmax=new GeomLines();
  GeomLines* edges=new GeomLines();
  all->add(new GeomMaterial(xmin, dk_red));
  all->add(new GeomMaterial(ymin, dk_green));
  all->add(new GeomMaterial(zmin, dk_blue));
  all->add(new GeomMaterial(xmax, lt_red));
  all->add(new GeomMaterial(ymax, lt_green));
  all->add(new GeomMaterial(zmax, lt_blue));
  all->add(new GeomMaterial(edges, gray));
  
  edges->add(Point(sbbox.min().x(), sbbox.min().y(), sbbox.min().z()), Point(sbbox.min().x(), sbbox.min().y(), sbbox.max().z()));
  edges->add(Point(sbbox.min().x(), sbbox.min().y(), sbbox.min().z()), Point(sbbox.min().x(), sbbox.max().y(), sbbox.min().z()));
  edges->add(Point(sbbox.min().x(), sbbox.min().y(), sbbox.min().z()), Point(sbbox.max().x(), sbbox.min().y(), sbbox.min().z()));
  edges->add(Point(sbbox.max().x(), sbbox.min().y(), sbbox.min().z()), Point(sbbox.max().x(), sbbox.max().y(), sbbox.min().z()));
  edges->add(Point(sbbox.max().x(), sbbox.min().y(), sbbox.min().z()), Point(sbbox.max().x(), sbbox.min().y(), sbbox.max().z()));
  edges->add(Point(sbbox.min().x(), sbbox.max().y(), sbbox.min().z()), Point(sbbox.max().x(), sbbox.max().y(), sbbox.min().z()));
  edges->add(Point(sbbox.min().x(), sbbox.max().y(), sbbox.min().z()), Point(sbbox.min().x(), sbbox.max().y(), sbbox.max().z()));
  edges->add(Point(sbbox.min().x(), sbbox.min().y(), sbbox.min().z()), Point(sbbox.min().x(), sbbox.min().y(), sbbox.max().z()));
  edges->add(Point(sbbox.min().x(), sbbox.min().y(), sbbox.max().z()), Point(sbbox.max().x(), sbbox.min().y(), sbbox.max().z()));
  edges->add(Point(sbbox.min().x(), sbbox.min().y(), sbbox.max().z()), Point(sbbox.min().x(), sbbox.max().y(), sbbox.max().z()));
  edges->add(Point(sbbox.max().x(), sbbox.max().y(), sbbox.min().z()), Point(sbbox.max().x(), sbbox.max().y(), sbbox.max().z()));
  edges->add(Point(sbbox.max().x(), sbbox.min().y(), sbbox.max().z()), Point(sbbox.max().x(), sbbox.max().y(), sbbox.max().z()));
  edges->add(Point(sbbox.min().x(), sbbox.max().y(), sbbox.max().z()), Point(sbbox.max().x(), sbbox.max().y(), sbbox.max().z()));
  int i;
  int nx=numx.get();
  int ny=numy.get();
  int nz=numz.get();
  for(i=0;i<nx;i++){
    double x=Interpolate(sbbox.min().x(), sbbox.max().x(), double(i+1)/double(nx+1));
    ymin->add(Point(x, sbbox.min().y(), sbbox.min().z()), Point(x, sbbox.min().y(), sbbox.max().z()));
    ymax->add(Point(x, sbbox.max().y(), sbbox.min().z()), Point(x, sbbox.max().y(), sbbox.max().z()));
    zmin->add(Point(x, sbbox.min().y(), sbbox.min().z()), Point(x, sbbox.max().y(), sbbox.min().z()));
    zmax->add(Point(x, sbbox.min().y(), sbbox.max().z()), Point(x, sbbox.max().y(), sbbox.max().z()));
  }
  for(i=0;i<ny;i++){
    double y=Interpolate(sbbox.min().y(), sbbox.max().y(), double(i+1)/double(ny+1));
    xmin->add(Point(sbbox.min().x(), y, sbbox.min().z()), Point(sbbox.min().x(), y, sbbox.max().z()));
    xmax->add(Point(sbbox.max().x(), y, sbbox.min().z()), Point(sbbox.max().x(), y, sbbox.max().z()));
    zmin->add(Point(sbbox.min().x(), y, sbbox.min().z()), Point(sbbox.max().x(), y, sbbox.min().z()));
    zmax->add(Point(sbbox.min().x(), y, sbbox.max().z()), Point(sbbox.max().x(), y, sbbox.max().z()));
  }
  for(i=0;i<nz;i++){
    double z=Interpolate(sbbox.min().z(), sbbox.max().z(), double(i+1)/double(nz+1));
    xmin->add(Point(sbbox.min().x(), sbbox.min().y(), z), Point(sbbox.min().x(), sbbox.max().y(), z));
    xmax->add(Point(sbbox.max().x(), sbbox.min().y(), z), Point(sbbox.max().x(), sbbox.max().y(), z));
    ymin->add(Point(sbbox.min().x(), sbbox.min().y(), z), Point(sbbox.max().x(), sbbox.min().y(), z));
    ymax->add(Point(sbbox.min().x(), sbbox.max().y(), z), Point(sbbox.max().x(), sbbox.max().y(), z));
  }
  ogeom->delAll();
  ogeom->addObj(all, "Field Cage");
}

} // End namespace SCIRun

