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
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/CurveField.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Modules/Visualization/StreamLines.h>
#include <Core/Containers/Handle.h>

#include <iostream>
#include <vector>
#include <math.h>

#include <Dataflow/share/share.h>

namespace SCIRun {

class PSECORESHARE StreamLines : public Module {
public:
  StreamLines(GuiContext* ctx);

  virtual ~StreamLines();

  virtual void execute();

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

  GuiDouble                     stepsize_;
  GuiDouble                     tolerance_;
  GuiInt                        maxsteps_;
  GuiInt                        direction_;
  GuiInt                        color_;
  GuiInt                        remove_colinear_;
  GuiInt                        method_;
  GuiInt                        np_;
};

DECLARE_MAKER(StreamLines)

StreamLines::StreamLines(GuiContext* ctx) : 
  Module("StreamLines", ctx, Source, "Visualization", "SCIRun"),
  vf_(0),
  sf_(0),
  stepsize_(ctx->subVar("stepsize")),
  tolerance_(ctx->subVar("tolerance")),
  maxsteps_(ctx->subVar("maxsteps")),
  direction_(ctx->subVar("direction")),
  color_(ctx->subVar("color")),
  remove_colinear_(ctx->subVar("remove-colinear")),
  method_(ctx->subVar("method")),
  np_(ctx->subVar("np"))
{
}

StreamLines::~StreamLines()
{
}


//! interpolate using the generic linear interpolator
static bool
interpolate(VectorFieldInterfaceHandle vfi, const Point &p, Vector &v)
{
  return vfi->interpolate(v, p) && (v.safe_normalize() > 0.0);
}


// LUTs for the RK-fehlberg algorithm 
static const double rkf_a[] =
  {16.0/135, 0, 6656.0/12825, 28561.0/56430, -9.0/50, 2.0/55};
static const double rkf_ab[] =
  {1.0/360, 0, -128.0/4275, -2197.0/75240, 1.0/50, 2.0/55};
//static const double rkf_c[] =
//  {0, 1.0/4, 3.0/8, 12.0/13, 1.0, 1.0/2}; // Not used, keep for reference.
static const double rkf_d[][5]=
  {{0, 0, 0, 0, 0},
   {1.0/4, 0, 0, 0, 0},
   {3.0/32, 9.0/32, 0, 0, 0},
   {1932.0/2197, -7200.0/2197, 7296.0/2197, 0, 0},
   {439.0/216, -8.0, 3680.0/513, -845.0/4104, 0},
   {-8.0/27, 2.0, -3544.0/2565, 1859.0/4104, -11.0/40}};

static int
ComputeRKFTerms(vector<Vector> &v, // storage for terms
		const Point &p,    // previous point
		double s,          // current step size
		VectorFieldInterfaceHandle vfi)
{
  // Already computed this one when we did the inside test.
  //  if (!interpolate(vfi, p, v[0]))
  //  {
  //    return -1;
  //  }
  v[0] *= s;
  
  if (!interpolate(vfi, p + v[0]*rkf_d[1][0], v[1]))
  {
    return 0;
  }
  v[1] *= s;
  
  if (!interpolate(vfi, p + v[0]*rkf_d[2][0] + v[1]*rkf_d[2][1], v[2]))
  {
    return 1;
  }
  v[2] *= s;
  
  if (!interpolate(vfi, p + v[0]*rkf_d[3][0] + v[1]*rkf_d[3][1] +
		   v[2]*rkf_d[3][2], v[3]))
  {
    return 2;
  }
  v[3] *= s;
  
  if (!interpolate(vfi, p + v[0]*rkf_d[4][0] + v[1]*rkf_d[4][1] +
		   v[2]*rkf_d[4][2] + v[3]*rkf_d[4][3], v[4]))
  {
    return 3;
  }
  v[4] *= s;
  
  if (!interpolate(vfi, p + v[0]*rkf_d[5][0] + v[1]*rkf_d[5][1] +
		   v[2]*rkf_d[5][2] + v[3]*rkf_d[5][3] +
		   v[4]*rkf_d[5][4], v[5]))
  {
    return 4;
  }
  v[5] *= s;

  return 5;
}


static void
FindRKF(vector<Point> &v, // storage for points
	Point x,          // initial point
	double t2,       // square error tolerance
	double s,         // initial step size
	int n,            // max number of steps
	VectorFieldInterfaceHandle vfi) // the field
{
  vector <Vector> terms(6, Vector(0.0, 0.0, 0.0));

  if (!interpolate(vfi, x, terms[0])) { return; }
  for (int i=0; i<n; i++)
  {
    // Compute the next set of terms.
    int tmp = ComputeRKFTerms(terms, x, s, vfi);
    if (tmp == -1)
    {
      break;
    }
    else if (tmp < 5)
    {
      s /= 1.5;
      continue;
    }

    // Compute the approximate local truncation error.
    const Vector err = terms[0]*rkf_ab[0] + terms[1]*rkf_ab[1]
      + terms[2]*rkf_ab[2] + terms[3]*rkf_ab[3] + terms[4]*rkf_ab[4]
      + terms[5]*rkf_ab[5];
    const double err2 = err.x()*err.x() + err.y()*err.y() + err.z()*err.z();
    
    // Is the error tolerable?  Adjust the step size accordingly.
    if (err2 * 16384.0 < t2)
    {
      s *= 2.0;
    }
    else if (err2 > t2)
    {
      s /= 2.0;
      continue;
    }

    // Compute and add the point to the list of points found.
    x = x  +  terms[0]*rkf_a[0] + terms[1]*rkf_a[1] + terms[2]*rkf_a[2] + 
      terms[3]*rkf_a[3] + terms[4]*rkf_a[4] + terms[5]*rkf_a[5];

    // If the new point is inside the field, add it.  Otherwise stop.
    if (!interpolate(vfi, x, terms[0])) { break; }
    v.push_back(x);
  }
}


static void
FindHeun(vector<Point> &v, // storage for points
	 Point x,          // initial point
	 double t2,       // square error tolerance
	 double s,         // initial step size
	 int n,            // max number of steps
	 VectorFieldInterfaceHandle vfi) // the field
{
  int i;
  Vector v0, v1;

  if (!interpolate(vfi, x, v0)) { return; }
  for (i=0; i < n; i ++)
  {
    v0 *= s;
    if (!interpolate(vfi, x + v0, v1)) { break; }
    v1 *= s;
    x += 0.5 * (v0 + v1);

    if (!interpolate(vfi, x, v0)) { break; }
    v.push_back(x);
  }
}


static void
FindRK4(vector<Point> &v,
	Point x,
	double t2,
	double s,
	int n,
	VectorFieldInterfaceHandle vfi)
{
  vector<Vector> terms(6, Vector(0.0, 0.0, 0.0));
  Vector f[4];
  int i;

  if (!interpolate(vfi, x, f[0])) { return; }
  for (i = 0; i < n; i++)
  {
    f[0] *= s;
    if (!interpolate(vfi, x + f[0] * 0.5, f[1])) { break; }
    f[1] *= s;
    if (!interpolate(vfi, x + f[1] * 0.5, f[2])) { break; }
    f[2] *= s;
    if (!interpolate(vfi, x + f[2], f[3])) { break; }
    f[3] *= s;

    x += (f[0] + 2.0 * f[1] + 2.0 * f[2] + f[3]) * (1.0 / 6.0);

    // If the new point is inside the field, add it.  Otherwise stop.
    if (!interpolate(vfi, x, f[0])) { break; }
    v.push_back(x);
  }
}


static void
FindAdamsBashforth(vector<Point> &v, // storage for points
		   Point x,          // initial point
		   double t2,       // square error tolerance
		   double s,         // initial step size
		   int n,            // max number of steps
		   VectorFieldInterfaceHandle vfi) // the field
{
  FindRK4(v, x, t2, s, Min(n, 5), vfi);
  if (v.size() < 5) { return; }
  Vector f[5];
  int i;

  for (i = 0; i < 5; i++)
  {
    interpolate(vfi, v[v.size() - 1 - i], f[i]);
  }
  x = v[v.size() - 1];

  for (i = 5; i < n; i++)
  {
    x += (s/720.) * (1901.0 * f[0] - 2774.0 * f[1] +
		     2616.0 * f[2] - 1274.0 * f[3] +
		     251.0 * f[4]);

    f[4] = f[3];
    f[3] = f[2];
    f[2] = f[1];
    f[1] = f[0];
    if (!interpolate(vfi, x, f[0])) { break; }
    v.push_back(x);
  }
}


void
StreamLinesCleanupPoints(vector<Point> &v, const vector<Point> &input,
			 double e2)
{
  unsigned int i, j;
  v.push_back(input[0]);
  j = 0;
  for (i=1; i < input.size()-1; i++)
  {
    const Vector v0 = input[i] - v[j];
    const Vector v1 = input[i] - input[i+1];
    if (Cross(v0, v1).length2() > e2 && Dot(v0, v1) < 0.0)
    {
      v.push_back(input[i]);
      j++;
    }
  }
  if (input.size() > 1) v.push_back(input[input.size()-1]);
}



void
StreamLinesAlgo::FindNodes(vector<Point> &v, // storage for points
			   Point x,          // initial point
			   double t2,       // square error tolerance
			   double s,         // initial step size
			   int n,            // max number of steps
			   VectorFieldInterfaceHandle vfi, // the field
			   bool remove_colinear_p,
			   int method)
{
  if (method == 0)
  {
    FindAdamsBashforth(v, x, t2, s, n, vfi);
  }
  else if (method == 1)
  {
    // TODO: Implement AdamsMoulton
  }
  else if (method == 2)
  {
    FindHeun(v, x, t2, s, n, vfi);
  }
  else if (method == 3)
  {
    FindRK4(v, x, t2, s, n, vfi);
  }
  else if (method == 4)
  {
    FindRKF(v, x, t2, s, n, vfi);
  }

  if (remove_colinear_p)
  {
    vector<Point> tmp;
    StreamLinesCleanupPoints(tmp, v, t2);
    v = tmp;
  }
}


static inline int
CLAMP(int a, int lower, int upper)
{
  if (a < lower) return lower;
  else if (a > upper) return upper;
  return a;
}



double
StreamLinesAccAlgo::RayPlaneIntersection(const Point &p, const Vector &dir,
					 const Point &p0, const Vector &pn)
{
  // Compute divisor.
  const double Vd = Dot(dir, pn);

  // Return no intersection if parallel to plane or no cross product.
  if (Vd < 1.0e-12) { return 1.0e24; }

  const double D = - Dot(pn, p0);

  const double V0 = - (Dot(pn, p) + D);
    
  return V0 / Vd;
}



void
StreamLines::execute()
{
  vfport_ = (FieldIPort*)get_iport("Flow field");
  sfport_ = (FieldIPort*)get_iport("Seeds");
  oport_=(FieldOPort*)get_oport("Streamlines");
  
  //must find vector field input port
  if (!vfport_) {
    error("Unable to initialize iport 'Flow field'.");
    return;
  }
   

  // must find seed field input port
  if (!sfport_) {
    error("Unable to initialize iport 'Seeds'.");
    return;
  }

  // must find output port
  if (!oport_) {
    error("Unable to initialize oport 'Streamlines'.");
    return;
  }
  
  // the vector field input is required
  if (!vfport_->get(vfhandle_) || !(vf_ = vfhandle_.get_rep())) {
    return;
  }
  
  // Check that the flow field input is a vector field.
  VectorFieldInterfaceHandle vfi = vf_->query_vector_interface(this);
  if (!vfi.get_rep()) {
    error("FlowField is not a Vector field.  Exiting.");
    return;
  }

  // the seed field input is required
  if (!sfport_->get(sfhandle_) || !(sf_ = sfhandle_.get_rep()))
    return;

  tolerance_.reset();
  double tolerance = tolerance_.get();
  stepsize_.reset();
  double stepsize = stepsize_.get();
  maxsteps_.reset();
  int maxsteps = maxsteps_.get();
  direction_.reset();
  int direction = direction_.get();
  color_.reset();
  int color = color_.get();

  if (method_.get() == 5 && vfhandle_->data_at() != Field::CELL)
  {
    error("The Cell Walk method only works for cell centered FlowFields.");
    return;
  }

  const TypeDescription *smtd = sf_->mesh()->get_type_description();
  const TypeDescription *sltd = sf_->data_at_type_description();
  if (method_.get() != 5)
  {
    CompileInfoHandle ci = StreamLinesAlgo::get_compile_info(smtd, sltd); 
    Handle<StreamLinesAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;
    vf_->mesh()->synchronize(Mesh::LOCATE_E);

    oport_->send(algo->execute(sf_->mesh(), vfi,
			       tolerance, stepsize, maxsteps, direction, color,
			       remove_colinear_.get(),
			       method_.get(), CLAMP(np_.get(), 1, 256)));
  }
  else
  {
    const TypeDescription *vtd = vfhandle_->get_type_description();
    CompileInfoHandle aci =
      StreamLinesAccAlgo::get_compile_info(smtd, sltd, vtd);
    Handle<StreamLinesAccAlgo> accalgo;
    if (!module_dynamic_compile(aci, accalgo)) return;
    vf_->mesh()->synchronize(Mesh::LOCATE_E);

    oport_->send(accalgo->execute(sf_->mesh(), vfhandle_, maxsteps,
				  direction, color,
				  remove_colinear_.get()));
  }
}


CompileInfoHandle
StreamLinesAlgo::get_compile_info(const TypeDescription *msrc,
				  const TypeDescription *sloc)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("StreamLinesAlgoT");
  static const string base_class_name("StreamLinesAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       msrc->get_filename() + "." +
		       sloc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
		       msrc->get_name() + ", " +
		       sloc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  msrc->fill_compile_info(rval);
  return rval;
}


CompileInfoHandle
StreamLinesAccAlgo::get_compile_info(const TypeDescription *msrc,
				     const TypeDescription *sloc,
				     const TypeDescription *vfld)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("StreamLinesAccAlgoT");
  static const string base_class_name("StreamLinesAccAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       msrc->get_filename() + "." +
		       sloc->get_filename() + "." +
		       vfld->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
		       msrc->get_name() + ", " +
		       sloc->get_name() + ", " +
		       vfld->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  msrc->fill_compile_info(rval);
  vfld->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun


