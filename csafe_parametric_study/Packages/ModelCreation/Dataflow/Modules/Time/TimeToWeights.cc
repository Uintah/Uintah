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
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>

#include <Dataflow/Network/Ports/MatrixPort.h>

namespace ModelCreation {

using namespace SCIRun;

class TimeToWeights : public Module {
public:
  TimeToWeights(GuiContext*);

  virtual void execute();

private:

  GuiDouble    slider_min_;
  GuiDouble    slider_max_;
  GuiDouble    range_max_;
  GuiDouble    range_min_;
  GuiString    playmode_;
  GuiString    dependence_;
  GuiDouble    current_;
  GuiString    execmode_;
  GuiInt       delay_;
  GuiDouble    inc_amount_;

  double    inc_;
  bool      loop_;
  bool      didrun_;

  double increment(double which, double lower, double upper);  
};


DECLARE_MAKER(TimeToWeights)
TimeToWeights::TimeToWeights(GuiContext* ctx)
  : Module("TimeToWeights", ctx, Source, "Time", "ModelCreation"),
    slider_min_(get_ctx()->subVar("slider_min")),
    slider_max_(get_ctx()->subVar("slider_max")),
    range_min_(get_ctx()->subVar("range_min")),
    range_max_(get_ctx()->subVar("range_max")),
    playmode_(get_ctx()->subVar("playmode")),
    dependence_(get_ctx()->subVar("dependence")),
    current_(get_ctx()->subVar("current")),
    execmode_(get_ctx()->subVar("execmode")),
    delay_(get_ctx()->subVar("delay")),
    inc_amount_(get_ctx()->subVar("inc-amount")),
    inc_(1),
    loop_(false),
    didrun_(false)
{
}

void TimeToWeights::execute()
{
  MatrixHandle TimeVector;
  MatrixHandle Time;
  MatrixHandle Weights;
  
  if (!(get_input_handle("TimeVector",TimeVector,true))) return;
  get_input_handle("Time",Time,false);
  
  if ((TimeVector->ncols() != 1)&&(TimeVector->nrows() != 1))
  {
    error("TimeToWeights: TimeVector needs to be vector (one of the dimesnions needs to be 1)");
    return;
  }
  
  
  double* times = TimeVector->get_data_pointer();
  int     size  = TimeVector->get_data_size();
  
  int start, middle, end;
  double starttime, middletime, endtime;
  
  int o1, o2;
  double w1, w2;
  
  start = 0;
  end = size-1;
  starttime = times[start];
  endtime = times[end];

  slider_min_.set(starttime);
  slider_max_.set(endtime);
  
  get_gui()->execute(get_id() + " update_range");
  reset_vars();
  
  bool senddata = true;
  double time;

  if (Time.get_rep())
  {
    if (!((Time->ncols() == 1)&&(Time->nrows() == 1)))
    {
      error("TimeToWeights: Time needs to be a scalar");
      return;
    }
    
    time = Time->get(0,0);
  }
  else
  {
    get_ctx()->reset();
    double startrange = range_min_.get();
    double endrange   = range_max_.get();
    
    double lower = startrange;
    double upper = endrange;
    
    if (lower > upper) {double tmp = lower; lower = upper; upper = tmp; }

    // Update the increment.
    if (playmode_.get() == "once" || playmode_.get() == "loop")
    {
      if (startrange>endrange) inc_ = -1.0; else inc_ = 1;
    }
    
    // If the current value is invalid, reset it to the start.
    if (current_.get() < lower || current_.get() > upper) 
    {
      current_.set(start);
      inc_ = (start>end)?-1:1;
    }

    // Cash execmode and reset it in case we bail out early.
    std::string execmode = execmode_.get();

    double current = current_.get();
 
    // If updating, we're done for now.
    if (didrun_ == false) 
    {
      senddata = true;
      didrun_ = true;
    } 
    else if (execmode == "step") 
    {
      current = increment(current, lower, upper);
      senddata = true;
    } 
    else if (execmode == "stepb") 
    {
      inc_ *= -1;
      current = increment(current, lower, upper);
      inc_ *= -1;
      senddata = true;
    } 
    else if (execmode == "play") 
    {
      if( !loop_ ) 
      {
        if (playmode_.get() == "once" && current >= endrange) current = startrange;
      }

      senddata = true;
      // User may have changed the execmode to stop so recheck.
      execmode_.reset();
      if ( loop_ = (execmode_.get() == "play") ) 
      {
        const int delay = delay_.get();
      
        if( delay > 0) 
        {
          const unsigned int secs = delay / 1000;
          const unsigned int msecs = delay % 1000;
          if (secs)  { sleep(secs); }
          if (msecs) { usleep(msecs * 1000); }
        }
    
        double next = increment(current, lower, upper);    

        // Incrementing may cause a stop in the execmode so recheck.
        execmode_.reset();
        if( loop_ = (execmode_.get() == "play") ) 
        {
          current = next;
          want_to_execute();
        }
      }
    } 
    else 
    {
      if( execmode == "rewind" ) current = startrange;
      else if( execmode == "fforward" )	current = endrange;
    
      senddata = true;
    
      if (playmode_.get() == "inc_w_exec") 
      {
        current = increment(current, lower, upper);
      }
    }
    
    current_.set(current);
    current_.reset();
  
    time = current;
  }
  
  if (time <= starttime)
  {
    o1 = 0; w1 = 1.0;
    o2 = -1; w2 = 0.0;
  }
  else if (time >= endtime)
  {
    o1 = size-1; w1 = 1.0;
    o2 = -1; w2 = 0.0;
  }
  else
  {
    while ((end-start) > 1)
    {
      middle = (end+start)/2;
      middletime = times[middle];
      
      if (time < middletime)
      {
        end = middle;
      }
      else if (time > middletime)
      {
        start = middle;
      }
      else
      {
        start = middle;
        break;
      }
    }
  
    starttime = times[start];
    endtime = times[end];
    if (time == starttime)
    {
      o1 = start; w1 = 1.0;
      o2 = -1; w2 = 0.0;
    }
    else if (time == endtime)
    {
      o1 = end; w1 = 1.0;
      o2 = -1; w2 = 0.0;
    }
    else
    {
      o1 = start; w1 = (endtime-time)/(endtime-starttime);
      o2 = end; w2 = (time-starttime)/(endtime-starttime);
    }
  }
    
  if (o2 == -1)
  {
    int* rr = scinew int[2];
    int* cc = scinew int[1];
    double* vv = scinew double[1];
    
    if ((rr==0)||(cc==0)||(vv==0))
    {
      if (rr) delete[] rr;
      if (cc) delete[] cc;
      if (vv) delete[] vv;     
      
      error("TimeToWeights: could not allocate sparse matrix");     
      return;
    }

    rr[0] = 0; rr[1] = 1; cc[0] = o1; vv[0] = w1;
    Weights = dynamic_cast<Matrix *>(scinew SparseRowMatrix(1,size,rr,cc,1,vv));
  }
  else
  {
    int* rr = scinew int[2];
    int* cc = scinew int[2];
    double* vv = scinew double[2];
    
    if ((rr==0)||(cc==0)||(vv==0))
    {
      if (rr) delete[] rr;
      if (cc) delete[] cc;
      if (vv) delete[] vv;     

      error("TimeToWeights: could not allocate sparse matrix");     
      return;
    }

    rr[0] = 0; rr[1] = 2; cc[0] = o1; cc[1] = o2; vv[0] = w1; vv[1] = w2;
    Weights = dynamic_cast<Matrix *>(scinew SparseRowMatrix(1,size,rr,cc,2,vv));  
  }

  if (Time.get_rep() == 0)
  {
    Time = dynamic_cast<Matrix *>(scinew DenseMatrix(1,1));
    if (Time.get_rep()) Time->put(0,0,time);
  }
  
  send_output_handle("TimeWeights",Weights,true);
  send_output_handle("Time",Time,true);
}



double TimeToWeights::increment(double current, double lower, double upper)
{
  // Do nothing if no range.
  if (upper == lower) {
    if (playmode_.get() == "once")
      execmode_.set( "stop" );
    return upper;
  }
  
  double inc_amount = inc_amount_.get();
  current += inc_ * inc_amount;

  if (current > upper) {
    if (playmode_.get() == "bounce1") 
    {
      inc_ *= -1.0;
      return increment(upper, lower, upper);
    } 
    else if (playmode_.get() == "bounce2") 
    {
      inc_ *= -1.0;
      return upper;
    } 
    else 
    {
      if (playmode_.get() == "once") execmode_.set( "stop" );
      return lower;
    }
  }
  if (current < lower) 
  {
    if (playmode_.get() == "bounce1") 
    {
      inc_ *= -1.0;
      return increment(lower, lower, upper);
    } 
    else if (playmode_.get() == "bounce2") 
    {
      inc_ *= -1.0;
      return lower;
    } 
    else 
    {
      if (playmode_.get() == "once")execmode_.set( "stop" );
      return upper;
    }
  }
  return current;
}


} // End namespace ModelCreation


