/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  StreamLines.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/ContourMesh.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Datatypes/MeshBase.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/ContourMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>

#include <iostream>
#include <vector>
#include <math.h>

#include <Dataflow/share/share.h>

namespace SCIRun {

using std::cerr;
using std::endl;

// LUTs for the RK-fehlberg algorithm 
double a[]   ={16.0/135, 0, 6656.0/12825, 28561.0/56430, -9.0/50, 2.0/55};
double ab[]  ={1.0/360, 0, -128.0/4275, -2197.0/75240, 1.0/50, 2.0/55};
double c[]   ={0, 1.0/4, 3.0/8, 12.0/13, 1.0, 1.0/2}; /* not used */
double d[][5]={{0, 0, 0, 0, 0},
	       {1.0/4, 0, 0, 0, 0},
	       {3.0/32, 9.0/32, 0, 0, 0},
	       {1932.0/2197, -7200.0/2197, 7296.0/2197, 0, 0},
	       {439.0/216, -8.0, 3680.0/513, -845.0/4104, 0},
	       {-8.0/27, 2.0, -3544.0/2565, 1859.0/4104, -11.0/40}};

class PSECORESHARE StreamLines : public Module {
public:
  StreamLines(const clString& id);

  virtual ~StreamLines();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);

private:
  // data members

  FieldIPort                    *vfport_;
  FieldIPort                    *sfport_;
  FieldOPort                    *oport_;

  FieldHandle                   vfhandle_;
  FieldHandle                   sfhandle_;
  FieldHandle                   ohandle_;

  Field                         *vf_;  // vector field
  Field                         *sf_;  // seed point field
  ContourField<double>          *cf_;

  ContourMesh                   *cmesh_;

  GenericInterpolate<Vector>    *interp_;

  GuiDouble                     stepsize_;
  GuiDouble                     tolerance_;
  GuiInt                        maxsteps_;

  //! loop through the nodes in the seed field
  template <class VectorField, class SeedField>
  void TemplatedExecute(VectorField *, SeedField *);

  //! find the nodes that make up a single stream line.
  //! This particular implementation uses Runge-Kutta-Fehlberg
  void FindStreamLineNodes(vector<Point>&, Point, float, float, int);

  //! compute the inner terms of the RKF formula
  int ComputeRKFTerms(vector<Vector>&,const Point&, float);
};

extern "C" PSECORESHARE Module* make_StreamLines(const clString& id) {
  return scinew StreamLines(id);
}

StreamLines::StreamLines(const clString& id)
  : Module("StreamLines", id, Source, "Visualization", "SCIRun"),
    stepsize_("stepsize",id,this),tolerance_("tolerance",id,this),
    maxsteps_("maxsteps",id,this)

{
  // these ports are now done automatically by the port manager

  //vfport_ = scinew FieldIPort(this, "FlowField", FieldIPort::Atomic);
  //add_iport(vfport_);

  //sfport_ = scinew FieldIPort(this, "SeedField", FieldIPort::Atomic);
  //add_iport(sfport_);

  //oport_ = scinew FieldOPort(this, "StreamLines", FieldIPort::Atomic);
  //add_oport(oport_);

  vf_ = 0;
  sf_ = 0;
  cf_ = 0;

  cmesh_ = 0;
}

StreamLines::~StreamLines()
{
}

int
StreamLines::ComputeRKFTerms(vector<Vector>& v /* storage for terms */,
			     const Point& p    /* previous point */,
			     float s           /* current step size */)
{
  int check = 0;

  check |= interp_->interpolate(p,v[0]);
  check |= interp_->interpolate(p+v[0]*d[1][0],v[1]);
  check |= interp_->interpolate(p+v[0]*d[2][0]+v[1]*d[2][1],v[2]);
  check |= interp_->interpolate(p+v[0]*d[3][0]+v[1]*d[3][1]+v[2]*d[3][2],v[3]);
  check |= interp_->interpolate(p+v[0]*d[4][0]+v[1]*d[4][1]+v[2]*d[4][2]+
				v[3]*d[4][3],v[4]);
  check |= interp_->interpolate(p+v[0]*d[5][0]+v[1]*d[5][1]+v[2]*d[5][2]+
				v[3]*d[5][3]+v[4]*d[5][4],v[5]);
  
  v[0]*=s;
  v[1]*=s;
  v[2]*=s;
  v[3]*=s;
  v[4]*=s;
  v[5]*=s;

  return check;
}
  
void
StreamLines::FindStreamLineNodes(vector<Point>& v /* storage for points */,
				 Point x          /* initial point */,
				 float t          /* error tolerance */,
				 float s          /* initial step size */,
				 int n            /* max number of steps */)
{
  int loop;
  vector<Vector> terms;
  Vector error;
  double err;
  Vector xv;

  terms.resize(6,0);

  // add the initial point to the list of points found.
  v.push_back(x);

  for (loop=0;loop<n;loop++) {

    // compute the next set of terms
    if (!ComputeRKFTerms(terms,x,s))
      break;

    // compute the approximate local truncation error
    error = terms[0]*ab[0]+terms[1]*ab[1]+terms[2]*ab[2]+
            terms[3]*ab[3]+terms[4]*ab[4]+terms[5]*ab[5];
    err = sqrt(error(0)*error(0)+error(1)*error(1)+error(2)*error(2));

    // is the error tolerable?  Adjust the step size accordingly.
    if (err<t/128.0)
      s*=2;
    else if (err>t) {
      s/=2;
      loop--;         // re-do this step.
      continue;
    }

    // compute and add the point to the list of points found.
    x = x + terms[0]*a[0]+terms[1]*a[1]+terms[2]*a[2]+
            terms[3]*a[3]+terms[4]*a[4]+terms[5]*a[5];

    // if the new point is inside the field, add it.  Otherwise stop.
    if (interp_->interpolate(x,xv))
      v.push_back(x);
    else
      break;
  }
}

template <class VectorField, class SeedField>
void StreamLines::TemplatedExecute(VectorField *vf, SeedField *sf)
{
  typedef typename VectorField::mesh_type        vf_mesh_type;
  typedef typename SeedField::mesh_type          sf_mesh_type;
  typedef typename vf_mesh_type::node_iterator   vf_node_iterator;
  typedef typename sf_mesh_type::node_iterator   sf_node_iterator;
  typedef typename ContourMesh::node_index       node_index;

  Point seed;
  Vector test;
  vector<Point> nodes;
  vector<Point>::iterator node_iter;
  node_index n1,n2;
  double tolerance;
  double stepsize;
  int maxsteps;

  sf_mesh_type *smesh =
    dynamic_cast<sf_mesh_type*>(sf->get_typed_mesh().get_rep());

  // try to find the streamline for each seed point
  sf_node_iterator seed_iter = smesh->node_begin();

  while (seed_iter!=smesh->node_end()) {

    // Is the seed point inside the field?
    smesh->get_point(seed,*seed_iter);
    if (!interp_->interpolate(seed,test))
      postMessage("StreamLines: WARNING: seed point "
		  "was not inside the field.");

    cerr << "new streamline." << endl;

    get_gui_doublevar(id,"tolerance",tolerance);
    get_gui_doublevar(id,"stepsize",stepsize);
    get_gui_intvar(id,"maxsteps",maxsteps);

    FindStreamLineNodes(nodes,seed,tolerance,stepsize,maxsteps);

    cerr << "done finding streamline." << endl;

    node_iter = nodes.begin();
    if (node_iter!=nodes.end())
      n1 = cmesh_->add_node(*node_iter);
    while (node_iter!=nodes.end()) {
      ++node_iter;
      if (node_iter!=nodes.end()) {
	n2 = cmesh_->add_node(*node_iter);
	cmesh_->add_edge(n1,n2);
	//cerr << "edge = " << n1 << " " << n2 << endl;
	n1 = n2;
	//fdata[index] = index++;
      }
    }

    cerr << "done adding streamline to contour field." << endl;

    ++seed_iter;
  }
}

void StreamLines::execute()
{
  // must find vector field input port
  if (!(vfport_=(FieldIPort*)get_iport(0)))
    return;

  // must find seed field input port
  if (!(sfport_=(FieldIPort*)get_iport(1)))
    return;

  // must find output port
  if (!(oport_=(FieldOPort*)get_oport(0)))
    return;

  // the vector field input is required
  if (!vfport_->get(vfhandle_) || !(vf_ = vfhandle_.get_rep()))
    return;

  // the seed field input is required
  if (!sfport_->get(sfhandle_) || !(sf_ = sfhandle_.get_rep()))
    return;

  // we expect that the flow field input is a vector field
  if (vf_->get_type_name(1) != "Vector") {
    postMessage("StreamLines: ERROR: FlowField is not a Vector field."
		"  Exiting.");
    return;
  }

  // might have to get Field::NODE
  cf_ = scinew ContourField<double>(Field::NONE);
  cmesh_ = dynamic_cast<ContourMesh*>(cf_->get_typed_mesh().get_rep());

  interp_ = (GenericInterpolate<Vector>*)vf_->query_interpolate();

  if (!interp_) {
    postMessage("StreamLines: ERROR: unable to locate an interpolation"
		" function for this field.  Exiting.");
    return;
  }

  cerr << "got here" << endl;

  // this is a pain...
  // use Marty's dispatch here instead...
  if (vf_->get_type_name(0) == "LatticeVol") {
    if (vf_->get_type_name(1) == "double") {
      if (sf_->get_type_name(-1) == "ContourField<double>") {
	TemplatedExecute((LatticeVol<double>*)vf_,(ContourField<double>*)sf_);
      } else if (sf_->get_type_name(-1) == "TriSurf<double>") {
	TemplatedExecute((LatticeVol<double>*)vf_,(TriSurf<double>*)sf_);
      }
    } else if (vf_->get_type_name(1) == "Vector") {
      if (sf_->get_type_name(-1) == "ContourField<double>") {
	TemplatedExecute((LatticeVol<Vector>*)vf_,(ContourField<double>*)sf_);
      } else if (sf_->get_type_name(-1) == "TriSurf<double>") {
	TemplatedExecute((LatticeVol<Vector>*)vf_,(TriSurf<double>*)sf_);
      }
    }
  } else if (vf_->get_type_name(0) =="TetVol") {
    if (vf_->get_type_name(1) == "double") {
      if (sf_->get_type_name(-1) == "ContourField<double>") {
	TemplatedExecute((TetVol<double>*)vf_,(ContourField<double>*)sf_);
      } else if (sf_->get_type_name(-1) == "TriSurf<double>") {
	TemplatedExecute((TetVol<double>*)vf_,(TriSurf<double>*)sf_);
      }
    } else if (vf_->get_type_name(1) == "Vector") {
      if (sf_->get_type_name(-1) == "ContourField<double>") {
	TemplatedExecute((TetVol<Vector>*)vf_,(ContourField<double>*)sf_);
      } else if (sf_->get_type_name(-1) == "TriSurf<double>") {
	TemplatedExecute((TetVol<Vector>*)vf_,(TriSurf<double>*)sf_);
      }
    }
  }

  oport_->send(cf_);
  cerr << "done with everything." << endl;
}

void StreamLines::tcl_command(TCLArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("StreamLines needs a minor command");
    return;
  }
 
  if (args[1] == "execute") {
    want_to_execute();
  } else {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace SCIRun


