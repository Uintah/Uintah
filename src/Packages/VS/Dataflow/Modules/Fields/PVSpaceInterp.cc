//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : PVSpaceInterp.cc
//    Author : Martin Cole
//    Date   : Thu Apr 21 09:54:23 2005

#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Runnable.h>
#include <Core/Util/Timer.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/TimePort.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Scheduler.h>

namespace VS {

using namespace SCIRun;
using std::cerr;
using std::endl;

class InterpController;

class PVSpaceInterp : public Module
{
public:
  PVSpaceInterp(GuiContext* ctx);
  virtual ~PVSpaceInterp();
  virtual void		execute();
  virtual void		tcl_command(GuiArgs& args, void*);
  
  bool check_consumed();
  void produce();
private:
  virtual void set_context(Scheduler* sched, Network* network);
  static bool consumed_callback(void *ths);
  void set_consumed(bool c) { consumed_ = c; }
  Point get_hip_params(unsigned &phase_mesh, unsigned &hip_idx);
  Point find_closest_hip(unsigned phase_idx, unsigned idx);
  void send_interp_field(const Point &p, unsigned phase_idx, unsigned hip_idx);
  void interp_field(const double *weights, const unsigned *interp_idxs);

  InterpController *		        runner_;
  Thread *		                runner_thread_;

  NrrdDataHandle                        hip_data_;
  NrrdDataHandle                        interp_space_;
  FieldHandle                           con_fld_;
  vector<FieldHandle>                   ppv_flds_;
  FieldHandle                           out_fld_;
  TimeViewerHandle                      time_viewer_h_;
  bool                                  consumed_;

  TimeIPort                            *time_port_;
  NrrdIPort                            *hip_port_;
  NrrdIPort                            *ispace_port_;
  FieldIPort                           *fld_port_;
  FieldOPort                           *out_port_;

  GuiDouble                             sample_rate_;
  GuiInt                                phase_idx_;
  GuiInt                                vol_idx_;
  GuiInt                                lvp_idx_;
  GuiInt                                rvp_idx_;
};


class InterpController : public Runnable {
public:
  InterpController(PVSpaceInterp* module, TimeViewerHandle tvh) : 
    module_(module), 
    throttle_(), 
    tvh_(tvh),
    dead_(0) 
  {};
  virtual ~InterpController();
  virtual void run();
  void set_dead(bool p) { dead_ = p; }
private:
  PVSpaceInterp            *module_;
  TimeThrottle	            throttle_;
  TimeViewerHandle          tvh_;
  bool		            dead_;
};

InterpController::~InterpController()
{
}

void
InterpController::run()
{
  throttle_.start();
  const double inc = 1./100.; // the rate at which we refresh.
  double t = throttle_.time();
  while (!dead_) {
    t = throttle_.time();
    throttle_.wait_for_time(t + inc);
    // Make sure downstream modules are ready to consume.
    if (! module_->check_consumed()) continue;
    module_->produce();
  }
}

DECLARE_MAKER(PVSpaceInterp) 

PVSpaceInterp::PVSpaceInterp(GuiContext* ctx) :
  Module("PVSpaceInterp", ctx, Filter, "Fields", "VS"),
  runner_(0),
  runner_thread_(0),
  hip_data_(0),
  interp_space_(0),
  con_fld_(0),
  out_fld_(0),
  time_viewer_h_(0),
  consumed_(true),
  time_port_(0),
  hip_port_(0),
  ispace_port_(0),
  fld_port_(0),
  out_port_(0),
  sample_rate_(ctx->subVar("sample_rate")),
  phase_idx_(ctx->subVar("phase_index")),
  vol_idx_(ctx->subVar("vol_index")),
  lvp_idx_(ctx->subVar("lvp_index")),
  rvp_idx_(ctx->subVar("rvp_index"))
{
}

PVSpaceInterp::~PVSpaceInterp()
{
  if (runner_thread_) {
    runner_->set_dead(true);
    runner_thread_->join();
    runner_thread_ = 0;
  }
}

void
PVSpaceInterp::set_context(Scheduler* sched, Network* network)
{
  Module::set_context(sched, network);
  sched->add_callback(consumed_callback, this);
}

bool
PVSpaceInterp::consumed_callback(void *ths)
{
  PVSpaceInterp* me = (PVSpaceInterp*)ths;
  me->set_consumed(true);
  return true;
}

bool
PVSpaceInterp::check_consumed() 
{
  // make sure downstream modules are ready for more data.
  return consumed_;
}

void 
PVSpaceInterp::produce() 
{
  consumed_ = false;
  unsigned phase_idx = 0;
  unsigned hip_idx = 0;
  Point p = get_hip_params(phase_idx, hip_idx);
  send_interp_field(p, phase_idx, hip_idx);
}

void 
PVSpaceInterp::send_interp_field(const Point &p, unsigned phase_idx, 
				 unsigned hip_idx) 
{

  FieldHandle fld = ppv_flds_[phase_idx];
  MeshHandle mesh_h = fld->mesh();
  HexVolMesh *mesh = (HexVolMesh*)mesh_h.get_rep();
  HexVolMesh::Node::array_type nodes;
  unsigned    interp_idxs[8];
  double      weights[8];
  
  //find closest...
  if (! mesh->get_weights(p, nodes, weights)) 
  {
    cerr << "Point " << p << " outside of mesh with phase index: " 
	 << phase_idx << std::endl;
    want_to_execute();
    return;
  }    
 
  //  cerr << "Point " << p << " interp with weights at phase mesh: " 
  //     << phase_idx << std::endl;
  interp_idxs[0] = phase_idx * 44 + nodes[0];
  interp_idxs[1] = phase_idx * 44 + nodes[1];
  interp_idxs[2] = phase_idx * 44 + nodes[2];
  interp_idxs[3] = phase_idx * 44 + nodes[3];
  interp_idxs[4] = phase_idx * 44 + nodes[4];
  interp_idxs[5] = phase_idx * 44 + nodes[5];
  interp_idxs[6] = phase_idx * 44 + nodes[6];
  interp_idxs[7] = phase_idx * 44 + nodes[7];

  interp_field(weights, interp_idxs);
  want_to_execute();
}

void 
PVSpaceInterp::interp_field(const double *weights, 
			    const unsigned *interp_idxs) 
{

  


  const int sz = interp_space_->nrrd->axis[1].size;
  float *dat = (float*)interp_space_->nrrd->data;
  float dst[sz][4];
  memset(dst, 0, 4 * sz * sizeof(float));
  for (int i = 0; i < 8; ++i) {
    //cerr << "weights[" << i << "]: " << weights[i] << std::endl;
    for(int j = 0; j < sz; ++j) {
      int idx = interp_idxs[i] * sz * 4;

      dst[j][0] += dat[idx + j * 4] * weights[i];
      dst[j][1] += dat[idx + j * 4 + 1] * weights[i];
      dst[j][2] += dat[idx + j * 4 + 2] * weights[i];
      dst[j][3] += dat[idx + j * 4 + 3] * weights[i];
    }
  }
  out_fld_ = con_fld_;
  out_fld_->generation++;
  //out_fld_.detach();
  //((HexVolField<double>*)out_fld_.get_rep())->mesh_detach();
  LockingHandle<HexVolMesh> mesh = 
    ((HexVolField<double>*)out_fld_.get_rep())->get_typed_mesh();
  vector<Point> &points = mesh->get_points();
  vector<double> &data = ((HexVolField<double>*)out_fld_.get_rep())->fdata();

  for (int i = 0; i < sz; ++i) {
    points[i].x(dst[i][0]);
    points[i].y(dst[i][1]);
    points[i].z(dst[i][2]);
    data[i] = dst[i][3];
  }
}


inline 
int
find_phase_index(float phase) 
{
  // phase centers from roy are:
  // 0.0375, 0.0750, 0.1250, 0.2000, 0.2750, 
  // 0.3500, 0.4250, 0.5000, 0.5750, 0.6500, 0.7000,
  // 0.8000, 0.9000, 1.0000

  const float phase_lookup[] = {0.01875, 0.05625, 0.1, 0.1625, 0.2375, 0.3125, 
				0.3875, 0.4625, 0.5375, 0.6125, 0.675, 0.75, 
				0.85, 0.95};
  int phase_mesh = 0;
  bool found = false;
  for (int i = 0; i < 13; ++i) 
  {
    if (phase > phase_lookup[i] && 
	phase <= phase_lookup[i + 1]) 
    {
      phase_mesh = i;
      found = true;
      break;
    }
  }
  if (! found) { phase_mesh = 13; }
  return phase_mesh;
}

Point
PVSpaceInterp::get_hip_params(unsigned &phase_mesh, unsigned &)
{
  // expected columns where data lives in HIP data.
  phase_idx_.reset();
  vol_idx_.reset();
  lvp_idx_.reset();
  rvp_idx_.reset();
  const unsigned phase_idx = phase_idx_.get();
  const unsigned vol_idx = vol_idx_.get();
  const unsigned lvp_idx = lvp_idx_.get();
  const unsigned rvp_idx = rvp_idx_.get();

  // Fetch the hip values for the current time.
  double abs_seconds = time_viewer_h_->view_elapsed_since_start();
  
  // Sample Rate HIP data is recorded at. 
  sample_rate_.reset();
  const double hip_sample_rate = sample_rate_.get();
  int idx = Round(abs_seconds * hip_sample_rate);
  if (idx > hip_data_->nrrd->axis[1].size - 1) {
    idx = hip_data_->nrrd->axis[1].size - 1;
  }

  float *dat = (float *)hip_data_->nrrd->data;
  int row = idx * hip_data_->nrrd->axis[0].size;
  float cur_phase = dat[row + phase_idx];

  phase_mesh = find_phase_index(cur_phase);

  // phase centers from roy are:
  const float phase_centers[] = { 0.0375, 0.0750, 0.1250, 0.2000, 0.2750, 
				  0.3500, 0.4250, 0.5000, 0.5750, 0.6500, 
				  0.7000, 0.8000, 0.9000, 1.0000 };

  const float center = phase_centers[phase_mesh];
  int f_idx = idx;
  float last_delta = fabs(cur_phase - center);
  if (last_delta > 0.0001) {
    float delta = last_delta + 1;
    // search forward...
    do {
      if (delta < last_delta) last_delta = delta;

      ++f_idx;
      if (f_idx > hip_data_->nrrd->axis[1].size - 1) {
	f_idx = hip_data_->nrrd->axis[1].size - 1;
	last_delta = delta;
	break;
      }
      row = f_idx * hip_data_->nrrd->axis[0].size;
      cur_phase = dat[row + phase_idx];
      delta = fabs(cur_phase - center);
    } while (delta < last_delta);

    float forward_delta = last_delta;
    int bk_idx = idx;
    row = bk_idx * hip_data_->nrrd->axis[0].size;
    cur_phase   = dat[row + phase_idx];
    last_delta = fabs(cur_phase - center);
    delta = last_delta + 1;
    
    // search backward
    do {
      if (delta < last_delta) last_delta = delta;
      --bk_idx;
      if (bk_idx < 0) {
	bk_idx = -1; // will add one below
	last_delta = delta;
	break; 
      }
      row = bk_idx * hip_data_->nrrd->axis[0].size;
      cur_phase = dat[row + phase_idx];
      delta = fabs(cur_phase - center);

    } while (delta < last_delta);

    // forward idx is one past the closest, backward is one before the closest.
    if (last_delta < forward_delta) idx = bk_idx + 1;
    else idx = f_idx - 1;

  }
  row = idx * hip_data_->nrrd->axis[0].size;
  float cur_lvp   = dat[row + lvp_idx];
  float cur_rvp   = dat[row + rvp_idx];
  float cur_vol   = dat[row + vol_idx];

  return Point(cur_vol, cur_lvp, cur_rvp);
}

void
PVSpaceInterp::execute()
{
  if (! time_port_) {
    time_port_ = (TimeIPort*)get_iport("Time");
    if (!time_port_)  {
      error("Unable to initialize iport Time.");
      return;
    }
  }
  time_port_->get(time_viewer_h_);
  if (time_viewer_h_.get_rep() == 0) {
    error("No data in the Time port. It is required.");
    return;
  }

  if (! hip_port_) {
    hip_port_ = (NrrdIPort*)get_iport("HIP Data");
    if (! hip_port_) {
      error("Unable to initialize iport 'HIP Data'.");
      return;
    }
  }
  hip_port_->get(hip_data_);
  if (! hip_data_.get_rep()) {
    error ("Unable to get input HIP Data.");
    return;
  } 

  if (! ispace_port_) {
    ispace_port_ = (NrrdIPort*)get_iport("Interpolation Space");
    if (! ispace_port_) {
      error("Unable to initialize iport 'Interpolation Space'.");
      return;
    }
  }
  ispace_port_->get(interp_space_);
  if (! interp_space_.get_rep()) {
    error ("Unable to get input Inerpolation Space data.");
    return;
  } 

  if (! fld_port_) {
    fld_port_ = (FieldIPort*)get_iport("Connectivity");
    if (! fld_port_) {
      error("Unable to initialize iport 'Connectivity'.");
      return;
    }
  }
  fld_port_->get(con_fld_);
  if (! con_fld_.get_rep()) {
    error ("Unable to get input Connectivity field.");
    return;
  } 

  if (ppv_flds_.size() == 0) {
    port_range_type range = get_iports("PPV Space");
    if (range.first == range.second) {
      error ("Need PPVspace fields.");
      return;
    }

    port_map_type::iterator pi = range.first;
    while (pi != range.second)
    {
      FieldIPort *ifield = (FieldIPort *)get_iport(pi->second);
      ++pi;
      FieldHandle fHandle;
      if (ifield->get(fHandle) && fHandle.get_rep())
      {
	ppv_flds_.push_back(fHandle);
      }
    }
    if (ppv_flds_.size() != 14) {
      error ("Need 14 PPVspace fields.");
      return;
    }
  }

  if (!runner_) {
    runner_ = scinew InterpController(this, time_viewer_h_);
    runner_thread_ = scinew Thread(runner_, string(id+" InterpController").c_str());
  }

  if (! out_port_) {
    out_port_ = (FieldOPort*)get_oport("Current");
    if (! out_port_) {
      error("Unable to initialize oport 'Current'.");
      return;
    }
  }
  out_port_->send(out_fld_);
}

void
PVSpaceInterp::tcl_command(GuiArgs& args, void* userdata) 
{
  if(args.count() < 2) {
    args.error("PVSpaceInterp needs a minor command");
    return;
  } else if(args[1] == "time") {
    // tmp
  } else {
    Module::tcl_command(args, userdata);
  }
}


} // End namespace VS
