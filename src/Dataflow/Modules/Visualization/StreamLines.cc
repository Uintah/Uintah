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
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/ContourField.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Modules/Visualization/StreamLines.h>

#include <iostream>
#include <vector>
#include <math.h>

#include <Dataflow/share/share.h>

namespace SCIRun {

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
  StreamLines(const string& id);

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

  GuiDouble                     stepsize_;
  GuiDouble                     tolerance_;
  GuiInt                        maxsteps_;

};

extern "C" PSECORESHARE Module* make_StreamLines(const string& id) {
  return scinew StreamLines(id);
}

StreamLines::StreamLines(const string& id) : 
  Module("StreamLines", id, Source, "Visualization", "SCIRun"),
  vf_(0),
  sf_(0),
  stepsize_("stepsize",id,this),
  tolerance_("tolerance",id,this),
  maxsteps_("maxsteps",id,this)  
{
}

StreamLines::~StreamLines()
{
}


//! interpolate using the generic linear interpolator
bool
StreamLinesAlgo::interpolate(VectorFieldInterface *vfi,
			     const Point &p, Vector &v)
{
  const bool b = vfi->interpolate(v,p);
  if (b && v.length2() > 0)
  {
    v.normalize(); // try not to skip cells - needs help from stepsize
  }
  return b;
}


bool
StreamLinesAlgo::ComputeRKFTerms(vector<Vector> &v, // storage for terms
				 const Point &p,    // previous point
				 double s,          // current step size
				 VectorFieldInterface *vfi)
{
  if (!interpolate(vfi, p, v[0]))
  {
    return false;
  }
  v[0] *= s;
  
  if (!interpolate(vfi, p + v[0]*d[1][0], v[1]))
  {
    return false;
  }
  v[1] *= s;
  
  if (!interpolate(vfi, p + v[0]*d[2][0] + v[1]*d[2][1], v[2]))
  {
    return false;
  }
  v[2] *= s;
  
  if (!interpolate(vfi, p + v[0]*d[3][0] + v[1]*d[3][1] + v[2]*d[3][2], v[3]))
  {
    return false;
  }
  v[3] *= s;
  
  if (!interpolate(vfi, p + v[0]*d[4][0] + v[1]*d[4][1] + v[2]*d[4][2] + 
		   v[3]*d[4][3], v[4]))
  {
    return false;
  }
  v[4] *= s;
  
  if (!interpolate(vfi, p + v[0]*d[5][0] + v[1]*d[5][1] + v[2]*d[5][2] + 
		   v[3]*d[5][3] + v[4]*d[5][4], v[5]))
  {
    return false;
  }
  v[5] *= s;

  return true;
}

  
void
StreamLinesAlgo::FindStreamLineNodes(vector<Point> &v, // storage for points
				     Point x,          // initial point
				     double t,          // error tolerance
				     double s,          // initial step size
				     int n,            // max number of steps
				     VectorFieldInterface *vfi) // the field
{
  int loop;
  vector <Vector> terms;
  Vector error;
  double err;
  Vector xv;

  terms.resize(6, Vector(0.0, 0.0, 0.0));

  // Add the initial point to the list of points found.
  v.push_back(x);

  for (loop=0; loop<n; loop++)
  {

    // Compute the next set of terms.
    if (!ComputeRKFTerms(terms, x, s, vfi))
    {
      break;
    }

    // Compute the approximate local truncation error.
    error = terms[0]*ab[0] + terms[1]*ab[1] + terms[2]*ab[2]
      + terms[3]*ab[3] + terms[4]*ab[4] + terms[5]*ab[5];
    err = sqrt(error(0)*error(0) + error(1)*error(1) + error(2)*error(2));
    
    // Is the error tolerable?  Adjust the step size accordingly.
    if (err < t/128.0)
    {
      s *= 2;
    }
    else if (err > t)
    {
      s /= 2;
      //loop--;         // Re-do this step.
      continue;
    }

    // Compute and add the point to the list of points found.
    x = x  +  terms[0]*a[0] + terms[1]*a[1] + terms[2]*a[2] + 
      terms[3]*a[3] + terms[4]*a[4] + terms[5]*a[5];

    // If the new point is inside the field, add it.  Otherwise stop.
    if (interpolate(vfi, x, xv))
    {
      v.push_back(x);
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
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
   

  // must find seed field input port
  if (!sfport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }

  // must find output port
  if (!oport_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
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

  // might have to get Field::NODE
  ContourField<double> *cf = scinew ContourField<double>(Field::NODE);
  ContourMeshHandle cmesh = cf->get_typed_mesh();

  double tolerance;
  double stepsize;
  int maxsteps;
  get_gui_doublevar(id, "tolerance", tolerance);
  get_gui_doublevar(id, "stepsize", stepsize);
  get_gui_intvar(id, "maxsteps", maxsteps);

  const TypeDescription *vmtd = vf_->mesh()->get_type_description();
  const TypeDescription *smtd = sf_->mesh()->get_type_description();
  const TypeDescription *sltd = sf_->data_at_type_description();
  CompileInfo *ci = StreamLinesAlgo::get_compile_info(vmtd, smtd, sltd); 
  DynamicAlgoHandle algo_handle;
  if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
  {
    cout << "Could not compile algorithm." << std::endl;
    return;
  }
  StreamLinesAlgo *algo =
    dynamic_cast<StreamLinesAlgo *>(algo_handle.get_rep());
  if (algo == 0)
  {
    cout << "Could not get algorithm." << std::endl;
    return;
  }
  algo->execute(vf_->mesh(), sf_->mesh(), vfi,
		tolerance, stepsize, maxsteps, cmesh);

  cf->resize_fdata();
  oport_->send(cf);
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


CompileInfo *
StreamLinesAlgo::get_compile_info(const TypeDescription *vmesh,
				  const TypeDescription *smesh,
				  const TypeDescription *sloc)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("StreamLinesAlgoT");
  static const string base_class_name("StreamLinesAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       vmesh->get_filename() + "." +
		       smesh->get_filename() + "." +
		       sloc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       vmesh->get_name() + ", " +
		       smesh->get_name() + ", " +
		       sloc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  vmesh->fill_compile_info(rval);
  smesh->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun


