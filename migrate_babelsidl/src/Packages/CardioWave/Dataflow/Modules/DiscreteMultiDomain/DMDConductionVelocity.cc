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

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>.
#include <Core/Algorithms/Converter/ConverterAlgo.h>

namespace CardioWave {

using namespace SCIRun;

class DMDConductionVelocity : public Module {
public:
  DMDConductionVelocity(GuiContext*);
  virtual void execute();

private:
  // for synchronous input
  int time_generation_;
  int potential1_generation_;
  int potential2_generation_;

  MatrixHandle Time_;
  MatrixHandle Potential1_;
  MatrixHandle Potential2_;
  
  GuiDouble guidistance_;
  GuiDouble guithreshold_;

  double time1_, time2_;
  bool foundtime1_, foundtime2_;
  double conductionvelocity_;
  bool startdetection_;

};

DECLARE_MAKER(DMDConductionVelocity)

DMDConductionVelocity::DMDConductionVelocity(GuiContext* ctx)
  : Module("DMDConductionVelocity", ctx, Source, "DiscreteMultiDomain", "CardioWave"),
  time_generation_(-1),
  potential1_generation_(-1),
  potential2_generation_(-1),
  guidistance_(ctx->subVar("distance")),
  guithreshold_(ctx->subVar("threshold")),
  time1_(0), time2_(0),
  foundtime1_(false), foundtime2_(false),
  conductionvelocity_(0),
  startdetection_(false)
{
}

void DMDConductionVelocity::execute()
{
  // Make sure we have synchronous inputs
  if ((Time_.get_rep() == 0)||(Time_->generation == time_generation_))
  {
    if (!(get_input_handle("Time",Time_,true))) return;
  }

  if ((Potential1_.get_rep() == 0)||(Potential1_->generation == potential1_generation_))
  {
    if (!(get_input_handle("Potential1",Potential1_,true))) return;
  }
  
  if ((Potential2_.get_rep() == 0)||(Potential2_->generation == potential2_generation_))
  {
    if (!(get_input_handle("Potential2",Potential2_,true))) return;
  }
  
  MatrixHandle Distance;
  MatrixHandle ActivationThreshold;
  MatrixHandle ConductionVelocity;
  
  get_input_handle("Distance",Distance,false);
  get_input_handle("Activation Threshold",ActivationThreshold,false);

  time_generation_ = Time_->generation;
  potential1_generation_ = Potential1_->generation;
  potential2_generation_ = Potential2_->generation;

  SCIRunAlgo::ConverterAlgo calgo(this);
  double time, potential1, potential2, distance, activationthreshold, conductionvelocity;

  if (Distance.get_rep()) if (calgo.MatrixToDouble(Distance,distance)) guidistance_.set(distance);
  if (ActivationThreshold.get_rep()) if (calgo.MatrixToDouble(ActivationThreshold,activationthreshold)) guithreshold_.set(activationthreshold);
  get_ctx()->reset();
  
  calgo.MatrixToDouble(Time_,time);
  calgo.MatrixToDouble(Potential1_,potential1);
  calgo.MatrixToDouble(Potential2_,potential2);
  
  distance = guidistance_.get();
  activationthreshold = guithreshold_.get();
  
  if (time1_ > time) { foundtime1_ = false; startdetection_ = false; }
  if (time2_ > time) { foundtime2_ = false; startdetection_ = false; }
  
  if ((potential1 < activationthreshold)&&(potential2 < activationthreshold)) startdetection_ = true;
  
  if (startdetection_)
  {
    if (foundtime1_ == false) if (potential1 > activationthreshold) { time1_ = time; foundtime1_ = true; }
    if (foundtime2_ == false) if (potential2 > activationthreshold) { time2_ = time; foundtime2_ = true; }
  }
  
  if (foundtime1_ && foundtime2_)
  {
    startdetection_ = false;
    conductionvelocity = distance/(time2_-time1_);
    
    if (calgo.DoubleToMatrix(conductionvelocity,ConductionVelocity)) send_output_handle("Conduction Velocity",ConductionVelocity,true);
  }
}

} // End namespace CardioWave


