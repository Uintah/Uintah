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
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Dataflow/Network/NetworkEditor.h>

#include <iostream>
#include <vector>
#include <math.h>

#include <Packages/Moulding/share/share.h>

namespace Moulding {

using namespace SCIRun;
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

class MouldingSHARE StreamLines : public Module {
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
  CurveField<double>          *cf_;

  GenericInterpolate<Vector>    *interp_;

  // member functions

  // find the nodes that make up a single stream line.
  // This particular implementation uses Runge-Kutta-Fehlberg
  void FindStreamLineNodes(vector<Point>&, Point, float, float, int);

  // compute the inner terms of the RKF formula
  int ComputeRKFTerms(vector<Vector>&,const Point&, float);
};

extern "C" MouldingSHARE Module* make_StreamLines(const clString& id) {
  return scinew StreamLines(id);
}

StreamLines::StreamLines(const clString& id)
  : Module("StreamLines", id, Source)
  //, "Visualization", "Moulding")
{
  vfport_ = scinew FieldIPort(this, "FlowField", FieldIPort::Atomic);
  add_iport(vfport_);

  sfport_ = scinew FieldIPort(this, "SeedField", FieldIPort::Atomic);
  add_iport(sfport_);

  oport_ = scinew FieldOPort(this, "StreamLines", FieldIPort::Atomic);
  add_oport(oport_);

  vf_ = 0;
  sf_ = 0;
  cf_ = 0;
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

void StreamLines::execute()
{
  CurveMesh *mesh;
  vector<Point> nodes;
  CurveMesh::Node::index_type n1,n2; 

  if (!vfport_->get(vfhandle_))
    return;

  if(!(vf_ = vfhandle_.get_rep()))
    return;

  // we expect that the field is a vector field
  if (vf_->get_type_name(1) != "Vector") {
    postMessage("StreamLines: ERROR: FlowField is not a Vector field."
		"  Exiting.");
    return;
  }

  cf_ = scinew CurveField<double>(Field::NODE);
  mesh = dynamic_cast<CurveMesh*>(cf_->get_typed_mesh().get_rep());
  CurveField<double>::fdata_type &fdata = cf_->fdata();

  interp_ = (GenericInterpolate<Vector>*)vf_->query_interpolate();

  if (!interp_) {
    postMessage("StreamLines: ERROR: unable to locate an interpolation"
		" function for this field.  Exiting.");
    return;
  }

  // get or generate seed_point(s) here
  vector<Point> seed_points;
  seed_points.push_back(Point(0.05,0.05,0.05));
  //seed_points.push_back(Point(-0.03,-0.03,-0.03));
  seed_points.push_back(Point(-0.05,-0.03,-0.05));

  // try to find the streamline for each seed point
  vector<Point>::iterator seed_iter = seed_points.begin();
  int index = 0;
  while (seed_iter!=seed_points.end()) {

    // Is the seed point inside the field?
    Vector test(0);
    if (!interp_->interpolate(*seed_iter,test))
      postMessage("StreamLines: WARNING: seed point "
		  "was not inside the field.");

    cerr << "new streamline." << endl;

    FindStreamLineNodes(nodes,*seed_iter,.001,.001,2000);

    cerr << "done finding streamline." << endl;

    fdata.resize(fdata.size()+nodes.size());

    vector<Point>::iterator node_iter = nodes.begin();
    if (node_iter!=nodes.end())
      n1 = mesh->add_node(*node_iter);
    while (node_iter!=nodes.end()) {
      ++node_iter;
      if (node_iter!=nodes.end()) {
	n2 = mesh->add_node(*node_iter);
	mesh->add_edge(n1,n2);
	//cerr << "edge = " << n1 << " " << n2 << endl;
	n1 = n2;
	fdata[index] = index++;
      }
    }

    cerr << "done adding streamline to contour field." << endl;

    ++seed_iter;
  }

  oport_->send(cf_);
  cerr << "done with everything." << endl;
}

void StreamLines::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Moulding


