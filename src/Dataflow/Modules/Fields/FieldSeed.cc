//static char *id="@(#) $Id$";

/*
 *  FieldSeed.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

/*
 *  This file seeds a volumetric mesh so you can do other stuff to it...
 *  it also spits out geometry for you...
 */

#include <SCICore/Util/Timer.h>
#include <SCICore/Containers/BitArray1.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomTri.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <SCICore/Datatypes/Mesh.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <SCICore/Datatypes/VectorField.h>
#include <SCICore/Datatypes/VectorFieldUG.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Multitask/ITC.h>
#include <SCICore/Multitask/Task.h>
#include <PSECore/Widgets/ScaledBoxWidget.h>

#include <stdlib.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace PSECore::Widgets;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::Containers;

class FieldSeed : public Module {
  ScalarFieldIPort *ifield;
  VectorFieldIPort *invector;
  GeometryOPort* ogeom;

  CrowdMonitor widget_lock;
  ScaledBoxWidget *widget;

  ScalarField*     osf;
  VectorField*     ovf;

  // tcl variables
  // number of sample points (multiplier times number of vertices)
  TCLdouble num_samps;   // number of samples
  TCLdouble samps_alpha; // alpha weight -> blending value
  TCLdouble hedge_scale; // scale for hedge hog
  TCLint    draw_pts;    //boolean for drawing point distribution
  TCLint    draw_vec;    // boolean for drawing all of the vectors
  TCLint    reghedge;    // 1 if you should use a regular sampling...
                         // changed - now controls

  int       cludge2d;    // 2D cludge -> slice data...
  TCLint    xstep;       // stepsize in X ^
  TCLint    ystep;       // stepsize in Y ^


  // these are internal variables
  
  Array1<int> nodes_in_widget; // nodes in the widget...
  int    points_id;
  int    hedge_id;

  double alpha;
  TexGeomLines *lines;

public:
  
  FieldSeed(const clString& id);
  virtual ~FieldSeed();
  virtual void execute();

  void tcl_command( TCLArgs&, void *);

  // other functions

  void Visualize(int which, int onoff);
  void DoElementAug(double);
  void ComputeWidgetNodes(void);
  
};


Module* make_FieldSeed(const clString& id) {
  return new FieldSeed(id);
}

FieldSeed::FieldSeed(const clString& id)
:Module("FieldSeed", id, Filter),widget(0),num_samps("num_samps",id,this),
 samps_alpha("samps_alpha",id,this),draw_pts("draw_pts",id,this),
 draw_vec("draw_vec",id,this),points_id(0),hedge_id(0),
 hedge_scale("hedge_scale",id,this),osf(0),ovf(0),//ug(0),vug(0),vrg(0),
 lines(0),alpha(1.0),
 reghedge("reghedge",id,this),
 xstep("xstep",id,this),ystep("ystep",id,this),cludge2d(0)
{

  ifield = scinew
    ScalarFieldIPort(this, "ScalarField", ScalarFieldIPort::Atomic);
  add_iport(ifield);

  // Create the input ports
  invector=scinew VectorFieldIPort(this, "Vector Field",
				  ScalarFieldIPort::Atomic);
  add_iport(invector);

  ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);
}

FieldSeed::~FieldSeed()
{
  // does anything need to be deleted?
}

void FieldSeed::execute()
{
  ScalarFieldHandle isf;

  // first see if we have a strucutured grid...

  VectorFieldHandle ivf;
  
  if (!invector->get(ivf)) // must have input vector field
    return; 

  if (ovf != ivf.get_rep()) { // vector pointer has changed...
    ovf = ivf.get_rep(); // assign the pointer...
  }
  
  VectorFieldRG* testg = ovf->getRG();
  
  if ((testg->nx == 1) ||
      (testg->ny == 1) ||
      (testg->nz == 1)) {
    cerr << "Doing a slice...\n";
    cludge2d = 1;
    return;
  }
  
  cludge2d = 0;

  if (!ifield->get(isf))
    return; // no input field yet

  if (osf != isf.get_rep()) { // pointers must match, or things have changed...
    osf = isf.get_rep();

    if (!widget) {
      widget = scinew ScaledBoxWidget(this, &widget_lock, 0.2);
      widget->SetCurrentMode(1);
      GeomObj *w=widget->GetWidget();
      ogeom->addObj(w, "That Guy", &widget_lock);
      widget->Connect(ogeom);
      widget->SetRatioR(0.2);
      widget->SetRatioD(0.2);
      widget->SetRatioI(0.2);
    }
 
    Point fmin,fmax;
    double scale = osf->longest_dimension()/30.0;
    osf->get_bounds(fmin,fmax);
    Point center = (fmin + fmax.vector())*0.5;
    
    //make this smaller
    
    double wscale=0.0;
    
    wscale = (fmax.x()-fmin.x())*0.15;
    
    Vector X(1*wscale,0,0),Y(0,1*wscale,0),Z(0,0,1*wscale);
    
    widget->SetScale(scale);
    widget->SetPosition(center,center+X,center+Y,center+Z);     

    // compute gradient magnitudes here...

    cerr << "Going to try and fill gradients...\n";
    cerr << "Now:\n";

    osf->fill_gradmags(); // just needs to be done once...
    cerr << "Done\n";
  }

  // this function just keeps this state curent...

}

// this is where the real dirt happens...

void FieldSeed::tcl_command(TCLArgs& args, void* userdata)
{
  if (args[1] == "init") { // initialize something...

    if (!osf) {
      execute(); // force an execute...
    }

    if (args[2] == "weights") { // compute distribution...
      num_samps.reset();
      int nsamps = num_samps.get(); // use it directly...
      
      cerr << nsamps << " " << num_samps.get() << "\n";

      cerr << "Doing compute samples...\n";

      osf->compute_samples(nsamps); // initial distribution only
    }
    if (args[2] == "alpha") {
      args[3].get_double(alpha);
      lines->alpha = alpha;
      ogeom->flushViews();
    }
    if (args[2] == "widget") { // augment distribution using widget...
      ComputeWidgetNodes();
      double w;
      args[3].get_double(w);
      DoElementAug(w);
    }
    if (args[2] == "grad_wt") { // augment distribution using gradient
      samps_alpha.reset();
      double alpha = samps_alpha.get();

      if (alpha < 1.0)
	osf->grad_augment(alpha,1-alpha);
    }
    if (args[2] == "grad_wt2") { // augment distribution using gradient
      samps_alpha.reset();
      double alpha = samps_alpha.get();

      if (alpha < 1.0)
	osf->hist_grad_augment(alpha,1-alpha);
    }
    if (args[2] == "dist_samp") { // distribute samples based on weights...
      osf->distribute_samples();
      draw_pts.reset();
      draw_vec.reset();
      if (points_id)
	Visualize(1,draw_pts.get());
      if (hedge_id)
	Visualize(0,draw_vec.get());
      cerr << "\nDid visualize\n";
    }
  } else if (args[1] == "draw") { // draw something...
    draw_pts.reset();
    draw_vec.reset();
    if (args[2] == "points") {
      Visualize(1,draw_pts.get());
    } else if (args[2] == "vectors") {
      Visualize(0,draw_vec.get());
    }
  } else {
    Module::tcl_command(args, userdata);
  }
}

// these functions deal with the 3D widget...
void FieldSeed::ComputeWidgetNodes()
{
  Point center,dx,dy,dz;
  
  widget_lock.read_lock();
  widget->GetPosition(center,dx,dy,dz);
  widget_lock.read_unlock();
  
  Vector X(dx-center),Y(dy-center),Z(dz-center); // for now...

  // it must be in 6 planes - use 2 corners

  Point corners[2];

  double d[3]; // dvalue for 3 planes
  Vector n[3]; // normals for 3 planes
  
  double testd[3]; // far extent of the planes...

  corners[0] = ((center - X) - Y) - Z;
  corners[1] = ((center + X) + Y) + Z;
 
  n[0] = Cross(X,Y);
  n[1] = Cross(Z,X);
  n[2] = Cross(Y,Z);
  
  testd[0] = (dz-center).length()*2; // distance for this guy...
  testd[1] = (dy-center).length()*2;
  testd[2] = (dx-center).length()*2;
  
  int i;
  for(i=0;i<3;i++) { // compute offsets as well...
    n[i].normalize();
    d[i] = -Dot(n[i],corners[0]);
  }

  nodes_in_widget.remove_all();  // eliminate all these guys

  // this only works for unstrucutred right now...

  ScalarFieldUG* ug = osf->getUG();

  if (!ug) {
    cerr << "Error - only works on unstructured right now!\n";
    return;
  }

  Mesh* ugmesh = ug->mesh.get_rep();


  for(i=0;i<ugmesh->nodes.size();i++) {
    Node *test = ugmesh->nodes[i].get_rep();

    if (test) {
      int j=0;

      for(;j<3;j++) {
	double dist = Dot(test->p,n[j])+d[j];
	if ((dist < 0) || (dist > testd[j])) 
	  j = 8;
      }

      if (j != 9) { // this means you passed...
	nodes_in_widget.add(i);
      }
    }
  }
}

void FieldSeed::DoElementAug(double weight)
{
  // this only works for unstrucutred right now...

  ScalarFieldUG* ug = osf->getUG();

  if (!ug) {
    cerr << "Error - only works on unstructured right now!\n";
    return;
  }

  Mesh* ugmesh = ug->mesh.get_rep();
  BitArray1 emask(ugmesh->elems.size(),0);

  for (int i=0;i<nodes_in_widget.size();i++) {
    for(int j=0;j<ugmesh->nodes[nodes_in_widget[i]]->elems.size();j++) {
      int test = ugmesh->nodes[nodes_in_widget[i]]->elems[j];
      if (!emask.is_set(test)) {
	emask.set(test); // flag it...

	osf->aug_elems[test].importance *= weight; // scale it
      }
    }
  }

  // the user should now click the redistribute button...

}

void FieldSeed::Visualize(int which, int onoff)
{

  if (cludge2d) {
    cerr << "Trying Hedge Hog! RG\n";
    
    if (hedge_id) {
      ogeom->delObj(hedge_id);
      ogeom->flushViews();
      hedge_id=0;
      lines=0;  // don't point to the data...
    }
    
    if (!onoff) { // shut it off
      return;
    }
    
    lines = scinew TexGeomLines;
    
    lines->alpha = alpha;  // set the alpha val...

    double glob_scale;
    
    hedge_scale.reset();
    
    glob_scale = hedge_scale.get(); // *ug->longest_dimension();
    
    //    cerr << glob_scale << " " << ug->longest_dimension() << " h\n";
    
    double minv=10000,maxv=-1.0;

    VectorFieldRG *vrg = ovf->getRG();
    
    int i;
    for(i=0;i<vrg->nx;i++) { // first get a handle on the magnitudes
      for(int j=0;j<vrg->ny;j++) {
        for(int k=0;k<vrg->nz;k++) {
          double mag = vrg->grid(i,j,k).length();
          if (mag < minv)
            minv = mag;
          if (mag > maxv)
            maxv = mag;
        }
      }
    }
    cerr << "First run through!\n";
    // now run though again
    
    double recipv = glob_scale/(maxv-minv);
    
    xstep.reset();
    ystep.reset();
    
    int dx = xstep.get();
    int dy = ystep.get();
    
    Point pmin,pmax;
    Vector one_cell;
    Point cur_p;

    vrg->get_bounds(pmin,pmax);
    
    one_cell = pmax-pmin;
    cur_p = pmin;
    one_cell.z(0.5*(pmin.z()+pmax.z()));
    
    for(i=0;i<vrg->nx;i+=dx) { // first get a handle on the magnitudes
      for(int j=0;j<vrg->ny;j+=dy) {
        for(int k=0;k<vrg->nz;k++) {
          Vector v = vrg->grid(i,j,k);
          double mag = (v.length()-minv)*recipv;
	  
          v.normalize();
	  
          if (mag != 0.0) { // 0 magnitude vectors don't count...
            v = v*mag;  // get the length set...
            
            double ddx = i/(vrg->nx-1.0);
            double ddy = j/(vrg->ny-1.0);
	    
            Point p(pmin.x() + one_cell.x()*ddx,
                    pmin.y() + one_cell.y()*ddy,
                    pmin.z());
	    
            lines->add(p-v,p+v,0.5/mag);
          }
	  
        }
      }
    }
    
    
    hedge_id = ogeom->addObj(lines,"Hedge Thing");
    
    ogeom->flushViews();
    
    cerr << "Done hedge! RG\n";
    return; // bolt if this was the case...
  }

  if (which == 1) { // doing the point cloud...
    if (points_id) {
      ogeom->delObj(points_id);
      ogeom->flushViews();
      points_id=0;
    }

    if (!onoff) { // shut it off
      return;
    }

    GeomPts *pts;

    pts = scinew GeomPts(osf->samples.size());

    for(int i=0;i<osf->samples.size();i++) {
      pts->add(osf->samples[i].loc);
    }

    points_id = ogeom->addObj(pts,"Sample Points");
    ogeom->flushViews();

  } else { // doing the hedgehogs...
    cerr << "Trying Hedge Hog!\n";

    if (hedge_id) {
      ogeom->delObj(hedge_id);
      ogeom->flushViews();
      hedge_id=0;
      lines=0;  // don't point to the data...
    }
    
    if (!onoff) { // shut it off
      return;
    }

    lines = scinew TexGeomLines;

    lines->alpha = alpha;  // set the alpha val...

    double glob_scale;

    hedge_scale.reset();

    glob_scale = hedge_scale.get()*osf->longest_dimension();

    cerr << glob_scale << " " << ovf->longest_dimension() << " h\n";
    
    int cache=0;
    
    double texval = 1.0/glob_scale;

    reghedge.reset();

    if (reghedge.get()) {
      Array1<Vector> vecs(osf->samples.size()); // cache them
      
      Array1<double> mags(osf->samples.size()); // magnitudes...
      double min=1111111,max=-1;

      int i;
      for(i=0;i<osf->samples.size();i++) {
	  //Vector oldv;

	ovf->interpolate(osf->samples[i].loc,vecs[i],cache);
	mags[i] = vecs[i].length();
	vecs[i].normalize(); // shared with above...
	
	if (mags[i] < min)
	  min = mags[i];
	if (mags[i] > max)
	  max = mags[i];
      }
      double rscale = 1.0/(max-min);
      
      for(i=0;i<osf->samples.size();i++) {
	Vector oldv = vecs[i]*(mags[i]-min)*glob_scale*rscale;
	lines->add(osf->samples[i].loc,osf->samples[i].loc+oldv,
		   1.0/((mags[i]-min)*glob_scale*rscale));
	
      }
      
      cerr << "Max: " << max << " Min: " << min << "\n";
    } else {
      for(int i=0;i<osf->samples.size();i++) {
        Vector oldv;
        ovf->interpolate(osf->samples[i].loc,oldv,cache);
        oldv.normalize();
        oldv = oldv * glob_scale;
        lines->add(osf->samples[i].loc,osf->samples[i].loc+oldv,
                   texval);
      }
    }

    hedge_id = ogeom->addObj(lines,"Hedge Thing TransParent"); // draw last...

    ogeom->flushViews();

    cerr << "Done hedge!\n";
  }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.5  1999/08/25 03:47:47  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:44  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:40  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:27  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:42  mcq
// Initial commit
//
// Revision 1.2  1999/04/28 20:51:10  dav
// deleted some files that are dependent on DaveW files
//
// Revision 1.1.1.1  1999/04/24 23:12:32  dav
// Import sources
//
//
