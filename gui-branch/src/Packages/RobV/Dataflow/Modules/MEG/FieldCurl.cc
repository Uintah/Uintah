/*
 *  FieldCurl.cc:  Unfinished modules
 *
 *  Written by:
 *   Robert Van Uitert
 *   Department of Computer Science
 *   University of Utah
 *   February 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Datatypes/VectorFieldRG.h>
#include <Core/Datatypes/VectorFieldUG.h>
#include <Core/GuiInterface/GuiVar.h>

namespace RobV {
using namespace SCIRun;

class FieldCurl : public Module {
  GuiInt interpolate;
  VectorFieldIPort* infield;
  VectorFieldOPort* outfield;
public:   
  FieldCurl(const clString& id);
  virtual ~FieldCurl();
  virtual void execute();
private:
  Vector curl(int x, int y, int z, VectorFieldRG* vf);
  Vector get_curl(Mesh *mesh, Element *elem,
		  Vector &v0, Vector &v1, Vector &v2, Vector &v3);
};

extern "C" Module* make_FieldCurl(const clString& id)
{
    return new FieldCurl(id);
}

FieldCurl::FieldCurl(const clString& id): Module("FieldCurl", id, Filter), interpolate("interpolate", id, this)
{
  infield=new VectorFieldIPort(this, "Vector", VectorFieldIPort::Atomic);
  add_iport(infield);

  // Create the output port
  outfield=new VectorFieldOPort(this, "Curl", VectorFieldIPort::Atomic);
  add_oport(outfield);
}

FieldCurl::~FieldCurl()
{
}

void FieldCurl::execute() {

  VectorFieldHandle vfIn;
  VectorFieldHandle vfOut;
  VectorFieldRG *vf;

  if(!infield->get(vfIn))
    return;
  VectorFieldUG* vfug=vfIn->getUG();
  VectorFieldRG* vfrg=vfIn->getRG();
  
  if (vfug) {   //unstructured grid
    VectorFieldUG* vfug2;
    if(interpolate.get()) {
      vfug2=new VectorFieldUG(VectorFieldUG::NodalValues);
      vfug2->mesh=vfug->mesh;
      vfug2->data.resize(vfug->data.size());
      Mesh* mesh=vfug->mesh.get_rep();
      int nnodes=mesh->nodes.size();
      Array1<Vector>& curls=vfug2->data;
      int i;
      
      for(i=0;i<nnodes;i++)
	curls[i]=Vector(0,0,0);
      
      int nelems=mesh->elems.size();
      for(i=0;i<nelems;i++) {
	if(i%1000 == 0)
	  update_progress(i, nelems);
	Element* e=mesh->elems[i];
	Vector v1=vfug->data[e->n[0]];
	Vector v2=vfug->data[e->n[1]];
	Vector v3=vfug->data[e->n[2]];
	Vector v4=vfug->data[e->n[3]];
	Vector curl = get_curl(mesh, e, v1, v2, v3, v4);
	for(int j=0;j<4;j++){
	     curls[e->n[j]]+=curl;
	}
       }
      for(i=0;i<nnodes;i++){
	const Node &n = mesh->node(i);
        curls[i]*=1./(n.elems.size());
      }
    } else {
      vfug2=new VectorFieldUG(VectorFieldUG::ElementValues);
      vfug2->mesh=vfug->mesh;
      Mesh* mesh=vfug->mesh.get_rep();
      int nelems=mesh->elems.size();
      vfug2->data.resize(nelems);
      for(int i=0;i<nelems;i++){
	if(i%10000 == 0)
	  update_progress(i, nelems);
	Element* e=mesh->elems[i];
	Vector v1=vfug->data[e->n[0]];
	Vector v2=vfug->data[e->n[1]];
	Vector v3=vfug->data[e->n[2]];
	Vector v4=vfug->data[e->n[3]];
	Vector curl = get_curl(mesh, e, v1, v2, v3, v4);
	vfug2->data[i]=curl;
      }
    }
    vfOut=vfug2;

    //end UG
  } else {   //regular grid
        int nx = vfrg->nx;
    int ny = vfrg->ny;
    int nz = vfrg->nz;
    vf=new VectorFieldRG();
    vf->resize(nx, ny, nz);
    Point min, max;
    vfrg->get_bounds(min, max);
    vf->set_bounds(min, max);

    for (int i=0; i<nx; i++) {
      update_progress(i, nx);
      for (int j=0; j<ny; j++) {
	for (int k=0; k<nz; k++) {
	  Vector rval = curl(i,j,k,vfrg);
	  vf->grid(i,j,k) = rval;
	}
      }
    }
    vfOut = vf;
  }
  outfield->send(vfOut);
}

//get curl for regular grid

Vector FieldCurl::curl(int x, int y, int z, VectorFieldRG* vf) {

    Point min, max;
    vf->get_bounds(min, max);
    Vector diagonal(max-min);
    int nx = vf->nx;
    int ny = vf->ny;
    int nz = vf->nz;
    
    double Bzy, Byz, Bzx, Bxz, Byx, Bxy;

    // this tries to use central differences...

    Vector h(0.5*(nx-1)/diagonal.x(),
	     0.5*(ny-1)/diagonal.y(),
	     0.5*(nz-1)/diagonal.z());
    // h is distances one over between nodes in each dimension...
    
   
    if (!x || (x == nx-1)) { // boundary...
      if (!x) {
	Bzx = ((vf->grid(x+1,y,z).z()-vf->grid(x,y,z).z())*2.0*h.x()); // end points are rare
	Byx = ((vf->grid(x+1,y,z).y()-vf->grid(x,y,z).y())*2.0*h.x());
      } else {
	Bzx = ((vf->grid(x,y,z).z()-vf->grid(x-1,y,z).z())*2.0*h.x());
	Byx = ((vf->grid(x,y,z).y()-vf->grid(x-1,y,z).y())*2.0*h.x());
      }
    } else { // just use central diferences...
      Bzx = ((vf->grid(x+1,y,z).z()-vf->grid(x-1,y,z).z())*h.x());
      Byx = ((vf->grid(x+1,y,z).y()-vf->grid(x-1,y,z).y())*h.x());
    }
    
    if (!y || (y == ny-1)) { // boundary...
      if (!y) {
	Bzy = ((vf->grid(x,y+1,z).z()-vf->grid(x,y,z).z())*2.0*h.y()); // end points are rare
	Bxy = ((vf->grid(x,y+1,z).x()-vf->grid(x,y,z).x())*2.0*h.y());
      } else {
	Bzy = ((vf->grid(x,y,z).z()-vf->grid(x,y-1,z).z())*2.0*h.y());
	Bxy = ((vf->grid(x,y,z).x()-vf->grid(x,y-1,z).x())*2.0*h.y());
      }
    } else { // just use central diferences...
      Bzy = ((vf->grid(x,y+1,z).z()-vf->grid(x,y-1,z).z())*h.y());
      Bxy = ((vf->grid(x,y+1,z).x()-vf->grid(x,y-1,z).x())*h.y());
    }
    
    if (!z || (z == nz-1)) { // boundary...
      if (!z) {
	Byz = ((vf->grid(x,y,z+1).y()-vf->grid(x,y,z).y())*2.0*h.z()); // end points are rare
	Bxz = ((vf->grid(x,y,z+1).x()-vf->grid(x,y,z).x())*2.0*h.z()); 
      } else {
	Byz = ((vf->grid(x,y,z).y()-vf->grid(x,y,z-1).y())*2.0*h.z());
	Bxz = ((vf->grid(x,y,z).x()-vf->grid(x,y,z-1).x())*2.0*h.z());
      }
    } else { // just use central diferences...
      Byz = ((vf->grid(x,y,z+1).y()-vf->grid(x,y,z-1).y())*h.z());
      Bxz = ((vf->grid(x,y,z+1).x()-vf->grid(x,y,z-1).x())*h.z());
    }

    Vector rval;

    rval.x(Bzy-Byz);
    rval.y(-Bzx+Bxz);
    rval.z(Byx-Bxy);
    return (rval);
}


//get curl for unstructured grid

Vector FieldCurl::get_curl(Mesh *mesh, Element *elem, Vector &v1, Vector &v2, Vector &v3, Vector &v4)
{

#ifndef STORE_ELEMENT_BASIS
    const Point &p1 = mesh->point(elem->n[0]);
    const Point &p2 = mesh->point(elem->n[1]);
    const Point &p3 = mesh->point(elem->n[2]);
    const Point &p4 = mesh->point(elem->n[3]);
    double x1=p1.x();
    double y1=p1.y();
    double z1=p1.z();
    double x2=p2.x();
    double y2=p2.y();
    double z2=p2.z();
    double x3=p3.x();
    double y3=p3.y();
    double z3=p3.z();
    double x4=p4.x();
    double y4=p4.y();
    double z4=p4.z();
    //double a1=+x2*(y3*z4-y4*z3)+x3*(y4*z2-y2*z4)+x4*(y2*z3-y3*z2);
    //double a2=-x3*(y4*z1-y1*z4)-x4*(y1*z3-y3*z1)-x1*(y3*z4-y4*z3);
    //double a3=+x4*(y1*z2-y2*z1)+x1*(y2*z4-y4*z2)+x2*(y4*z1-y1*z4);
    //double a4=-x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1);
    //double iV6=1./(a1+a2+a3+a4);

    double b1=-(y3*z4-y4*z3)-(y4*z2-y2*z4)-(y2*z3-y3*z2);
    double c1=+(x3*z4-x4*z3)+(x4*z2-x2*z4)+(x2*z3-x3*z2);
    double d1=-(x3*y4-x4*y3)-(x4*y2-x2*y4)-(x2*y3-x3*y2);

    double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
    double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
    double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);

    double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
    double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
    double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);

    double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
    double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
    double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);

#else

    double b1=elem->g[0].x();
    double c1=elem->g[0].y();
    double d1=elem->g[0].z();
    
    double b2=elem->g[1].x();
    double c2=elem->g[1].y();
    double d2=elem->g[1].z();
    
    double b3=elem->g[2].x();
    double c3=elem->g[2].y();
    double d3=elem->g[2].z();
    
    double b4=elem->g[3].x();
    double c4=elem->g[3].y();
    double d4=elem->g[3].z();

#endif

    double Bx = (v1.z()*c1+v2.z()*c2+v3.z()*c3+v4.z()*c4) - (v1.y()*d1+v2.y()*d2+v3.y()*d3+v4.y()*d4);
    double By = -(v1.z()*b1+v2.z()*b2+v3.z()*b3+v4.z()*b4) + (v1.x()*d1+v2.x()*d2+v3.x()*d3+v4.x()*d4);
    double Bz = (v1.y()*b1+v2.y()*b2+v3.y()*b3+v4.y()*b4) - (v1.x()*c1+v2.x()*c2+v3.x()*c3+v4.x()*c4);
    
    Vector curl(Bx, By, Bz);
    return(curl);

} // End namespace RobV
}

