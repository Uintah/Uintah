/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

namespace SCIRun {

class StreamLines : public Module {
public:
  StreamLines(GuiContext* ctx);

  virtual ~StreamLines();

  virtual void execute();

private:
  GuiDouble                     gStepsize_;
  GuiDouble                     gTolerance_;
  GuiInt                        gMaxsteps_;
  GuiInt                        gDirection_;
  GuiInt                        gColor_;
  GuiInt                        gRemove_Colinear_;
  GuiInt                        gMethod_;
  GuiInt                        gNp_;

  double tolerance_;
  double stepsize_;
  int maxsteps_;
  int direction_;
  int color_;
  int remove_colinear_;
  int method_;
  int np_;

  FieldHandle fHandle_;

  int vfGeneration_;
  int sfGeneration_;

  bool error_;
};

DECLARE_MAKER(StreamLines)

StreamLines::StreamLines(GuiContext* ctx) : 
  Module("StreamLines", ctx, Source, "Visualization", "SCIRun"),
  gStepsize_(ctx->subVar("stepsize")),
  gTolerance_(ctx->subVar("tolerance")),
  gMaxsteps_(ctx->subVar("maxsteps")),
  gDirection_(ctx->subVar("direction")),
  gColor_(ctx->subVar("color")),
  gRemove_Colinear_(ctx->subVar("remove-colinear")),
  gMethod_(ctx->subVar("method")),
  gNp_(ctx->subVar("np")),
  vfGeneration_(-1),
  sfGeneration_(-1),
  error_(0)
{
}

StreamLines::~StreamLines()
{
}


//! interpolate using the generic linear interpolator
static inline bool
interpolate(const VectorFieldInterfaceHandle &vfi, const Point &p, Vector &v)
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
ComputeRKFTerms(Vector v[6],       // storage for terms
		const Point &p,    // previous point
		double s,          // current step size
		const VectorFieldInterfaceHandle &vfi)
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
	const VectorFieldInterfaceHandle &vfi) // the field
{
  Vector terms[6];

  if (!interpolate(vfi, x, terms[0])) { return; }
  for (int i=0; i<n; i++)
  {
    // Compute the next set of terms.
    if (ComputeRKFTerms(terms, x, s, vfi) < 5)
    {
      s /= 1.5;
      continue;
    }

    // Compute the approximate local truncation error.
    const Vector err = terms[0]*rkf_ab[0] + terms[1]*rkf_ab[1]
      + terms[2]*rkf_ab[2] + terms[3]*rkf_ab[3] + terms[4]*rkf_ab[4]
      + terms[5]*rkf_ab[5];
    const double err2 = err.length2();
    
    // Is the error tolerable?  Adjust the step size accordingly.  Too
    // small?  Grow it for next time but keep small-error result.  Too
    // big?  Recompute with smaller size.
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
	 const VectorFieldInterfaceHandle &vfi) // the field
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
	const VectorFieldInterfaceHandle &vfi)
{
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
		   double t2,        // square error tolerance
		   double s,         // initial step size
		   int n,            // max number of steps
		   const VectorFieldInterfaceHandle &vfi) // the field
{
  FindRK4(v, x, t2, s, Min(n, 5), vfi);
  if (v.size() < 5) {
    cerr << "Streamlines - FindAdamsBashforth: Ending early, less than 5 points.\n\n";
    return;
  }

  Vector f[5];
  int i;

  for (i = 0; i < 5; i++)
    interpolate(vfi, v[v.size() - 1 - i], f[i]);
  
  x = v[v.size() - 1];
  
  for (i = 5; i < n; i++) {
    x += (s/720.) * (1901.0 * f[0] - 2774.0 * f[1] +
		     2616.0 * f[2] - 1274.0 * f[3] +
		     251.0 * f[4]);

    f[4] = f[3];
    f[3] = f[2];
    f[2] = f[1];
    f[1] = f[0];

    if (!interpolate(vfi, x, f[0])) {
      cerr << "Streamlines - FindAdamsBashforth: Ending early, can not interpolate\n";
      cerr << "vfi interpolate returned " << vfi->interpolate(f[0], x) << "\n";
      cerr << "Point " << x.x() << "  " << x.y() << "  " << x.z() << "\n";
      cerr << "Value " << f[0].x() << "  " << f[0].y() << "  " << f[0].z() << "\n";
      cerr << "safe_normalize returned " << f[0].safe_normalize() << "\n\n";

      break; 
    }

    v.push_back(x);
  }
}


vector<Point>::iterator
StreamLinesCleanupPoints(vector<Point> &input, double e2)
{
  unsigned int i, j;
  j = 0;
  for (i=1; i < input.size()-1; i++)
  {
    const Vector v0 = input[i] - input[j];
    const Vector v1 = input[i] - input[i+1];
    if (Cross(v0, v1).length2() > e2 && Dot(v0, v1) < 0.0)
    {
      j++;
      if (i != j) { input[j] = input[i]; }
    }
  }
  if (input.size() > 1) { j++; input[j] = input[input.size()-1]; }
  return input.begin() + j + 1;
}


void
StreamLinesAlgo::FindNodes(vector<Point> &v, // storage for points
			   Point x,          // initial point
			   double t2,       // square error tolerance
			   double s,         // initial step size
			   int n,            // max number of steps
			   const VectorFieldInterfaceHandle &vfi, // the field
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
    v.erase(StreamLinesCleanupPoints(v, t2), v.end());
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
  update_state(NeedData);
 
  FieldIPort* vfport = (FieldIPort*)get_iport("Flow field");
  FieldHandle vfHandle;
  Field *vField;  // vector field
  
  // the vector field input is required
  if (!vfport->get(vfHandle) || !(vField = vfHandle.get_rep())) {
    error( "No Vector field handle or representation" );
    return;
  }
  
  // Check that the flow field input is a vector field.
  VectorFieldInterfaceHandle vfi = vField->query_vector_interface(this);
  if (!vfi.get_rep()) {
    error("FlowField is not a Vector field.");
    return;
  }

  // TODO:  Fix this for cell walking on surfaces.
  if (vField->mesh()->dimensionality() != 3)
  {
    error("The StreamLines module currently only works for volume fields.");
    return;
  }

  FieldIPort* sfport = (FieldIPort*)get_iport("Seeds");
  FieldHandle sfHandle;
  Field *sField;  // seed point field

  // the seed field input is required
  if (!sfport->get(sfHandle) || !(sField = sfHandle.get_rep())) {
    error( "No Seed field handle or representation" );
    return;
  }

  bool update = false;

  // Check to see if the input field has changed.
  if( vfGeneration_ != vfHandle->generation )
  {
    vfGeneration_ = vfHandle->generation;
    update = true;
  }

  // Check to see if the input field has changed.
  if( sfGeneration_ != sfHandle->generation )
  {
    sfGeneration_ = sfHandle->generation;
    update = true;
  }

  double tolerance = gTolerance_.get();
  double stepsize = gStepsize_.get();
  int maxsteps = gMaxsteps_.get();
  int direction = gDirection_.get();
  int color = gColor_.get();
  int remove_colinear = gRemove_Colinear_.get();
  int method = gMethod_.get();
  int np = gNp_.get();
  
  if( tolerance_ != tolerance ||
      stepsize_  != stepsize  ||
      maxsteps_  != maxsteps  ||
      direction_ != direction ||
      color_     != color ||
      remove_colinear_ != remove_colinear  ||
      method_    != method  ||
      np_        != np ) {

    tolerance_ = tolerance;
    stepsize_  = stepsize;
    maxsteps_  = maxsteps;
    direction_ = direction;
    color_     = color;
    remove_colinear_ = remove_colinear;
    method_    = method;
    np_        = np;

    update = true;
  }

  if( !fHandle_.get_rep() ||
      update ||
      error_ ) {

    update_state(JustStarted);

    error_ = false;

    const TypeDescription *smtd = sField->mesh()->get_type_description();
    const TypeDescription *sltd = sField->order_type_description();

    if (method_ == 5 ) {

      if( vfHandle->basis_order() != 0) {
	error("The Cell Walk method only works for cell centered FlowFields.");
	error_ = true;
	return;
      }

      const TypeDescription *vtd = vfHandle->get_type_description();
      CompileInfoHandle aci =
	StreamLinesAccAlgo::get_compile_info(smtd, sltd, vtd);
      Handle<StreamLinesAccAlgo> accalgo;
      if (!module_dynamic_compile(aci, accalgo)) return;
      vField->mesh()->synchronize(Mesh::LOCATE_E);
      
      fHandle_ = accalgo->execute(sField->mesh(), vfHandle, maxsteps,
				  direction, color,
				  remove_colinear);
    } else {
      CompileInfoHandle ci = StreamLinesAlgo::get_compile_info(smtd, sltd); 
      Handle<StreamLinesAlgo> algo;
      if (!module_dynamic_compile(ci, algo)) return;
      vField->mesh()->synchronize(Mesh::LOCATE_E);
      
      fHandle_ = algo->execute(sField->mesh(), vfi,
			       tolerance, stepsize, maxsteps,
			       direction, color,
			       remove_colinear,
			       method, CLAMP(np, 1, 256));
    }

    update_state(Completed);
  }
   
  // Send the data downstream
  if( fHandle_.get_rep() )
  {
    FieldOPort *ofield_port = (FieldOPort *) get_oport("Streamlines");
    ofield_port->send( fHandle_ );
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


