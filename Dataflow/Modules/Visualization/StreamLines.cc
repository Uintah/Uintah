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

// LUTs for the RK-fehlberg algorithm 
const double a[]  ={16.0/135, 0, 6656.0/12825, 28561.0/56430, -9.0/50, 2.0/55};
const double ab[]  ={1.0/360, 0, -128.0/4275, -2197.0/75240, 1.0/50, 2.0/55};
  //const double c[]   ={0, 1.0/4, 3.0/8, 12.0/13, 1.0, 1.0/2}; /* not used */
const double d[][5]={{0, 0, 0, 0, 0},
		     {1.0/4, 0, 0, 0, 0},
		     {3.0/32, 9.0/32, 0, 0, 0},
		     {1932.0/2197, -7200.0/2197, 7296.0/2197, 0, 0},
		     {439.0/216, -8.0, 3680.0/513, -845.0/4104, 0},
		     {-8.0/27, 2.0, -3544.0/2565, 1859.0/4104, -11.0/40}};

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
  remove_colinear_(ctx->subVar("remove-colinear"))
{
}

StreamLines::~StreamLines()
{
}


//! interpolate using the generic linear interpolator
static bool
interpolate(VectorFieldInterface *vfi,
	    const Point &p, Vector &v)
{
  if (vfi->interpolate(v,p))
  {
    if (v.safe_normalize() > 0.0)
    {
      return true;
    }
  }
  return false;
}


static int
ComputeRKFTerms(vector<Vector> &v, // storage for terms
		const Point &p,    // previous point
		double s,          // current step size
		VectorFieldInterface *vfi)
{
  if (!interpolate(vfi, p, v[0]))
  {
    return -1;
  }
  v[0] *= s;
  
  if (!interpolate(vfi, p + v[0]*d[1][0], v[1]))
  {
    return 0;
  }
  v[1] *= s;
  
  if (!interpolate(vfi, p + v[0]*d[2][0] + v[1]*d[2][1], v[2]))
  {
    return 1;
  }
  v[2] *= s;
  
  if (!interpolate(vfi, p + v[0]*d[3][0] + v[1]*d[3][1] + v[2]*d[3][2], v[3]))
  {
    return 2;
  }
  v[3] *= s;
  
  if (!interpolate(vfi, p + v[0]*d[4][0] + v[1]*d[4][1] + v[2]*d[4][2] + 
		   v[3]*d[4][3], v[4]))
  {
    return 3;
  }
  v[4] *= s;
  
  if (!interpolate(vfi, p + v[0]*d[5][0] + v[1]*d[5][1] + v[2]*d[5][2] + 
		   v[3]*d[5][3] + v[4]*d[5][4], v[5]))
  {
    return 4;
  }
  v[5] *= s;

  return 5;
}



void
StreamLinesAlgo::FindStreamLineNodes(vector<Point> &v, // storage for points
				     Point x,          // initial point
				     double t2,       // square error tolerance
				     double s,         // initial step size
				     int n,            // max number of steps
				     VectorFieldInterface *vfi, // the field
				     bool remove_colinear_p)
{
  vector <Vector> terms(6, Vector(0.0, 0.0, 0.0));

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
    const Vector err = terms[0]*ab[0] + terms[1]*ab[1] + terms[2]*ab[2]
      + terms[3]*ab[3] + terms[4]*ab[4] + terms[5]*ab[5];
    const double err2 = err.x()*err.x() + err.y()*err.y() + err.z()*err.z();
    
    // Is the error tolerable?  Adjust the step size accordingly.
    if (err2 * 16384.0 < t2) // err < t/128.0
    {
      s *= 2;
    }
    else if (err2 > t2)
    {
      s /= 2;
      continue;
    }

    // Compute and add the point to the list of points found.
    x = x  +  terms[0]*a[0] + terms[1]*a[1] + terms[2]*a[2] + 
      terms[3]*a[3] + terms[4]*a[4] + terms[5]*a[5];

    // If the new point is inside the field, add it.  Otherwise stop.
    Vector xv;
    if (vfi->interpolate(xv, x))
    {
      if (remove_colinear_p && v.size() > 1)
      {
	const Vector a = v[v.size()-2] - v[v.size()-1];
	const Vector b = x - v[v.size()-1];
	if (Cross(a, b).length2() > 1.0e-12 && Dot(a, b) < 0.0)
	{
	  // Not colinear, push.
	  v.push_back(x);
	}
	else
	{
	  // Colinear, replace.
	  v[v.size()-1] = x;
	}
      }
      else
      {
	v.push_back(x);
      }
    }
    else
    {
      break;
    }
  }
}



void StreamLines::execute()
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
  VectorFieldInterface *vfi = vf_->query_vector_interface();
  if (!vfi) {
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

  const TypeDescription *smtd = sf_->mesh()->get_type_description();
  const TypeDescription *sltd = sf_->data_at_type_description();
  CompileInfo *ci = StreamLinesAlgo::get_compile_info(smtd, sltd); 
  Handle<StreamLinesAlgo> algo;
  if (!module_dynamic_compile(*ci, algo)) return;
  vf_->mesh()->synchronize(Mesh::LOCATE_E);

  oport_->send(algo->execute(sf_->mesh(), vfi,
			     tolerance, stepsize, maxsteps, direction, color,
			     remove_colinear_.get()));
}


CompileInfo *
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


} // End namespace SCIRun


