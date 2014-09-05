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
 * FILE: TimeDataReader.cc
 * AUTH: Jeroen G Stinstra
 * DATE: 23 MAR 2005
 */
 

#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/NrrdString.h>

#include <Packages/CardioWave/Core/Datatypes/TimeDataFile.h>

namespace CardioWave {

using namespace SCIRun;

class TimeDataReader : public Module {
  GuiString row_or_col_;
  GuiDouble selectable_min_;
  GuiDouble selectable_max_;
  GuiInt    selectable_inc_;
  GuiString selectable_units_;
  GuiInt    range_min_;
  GuiInt    range_max_;
  GuiString playmode_;
  GuiString dependence_;
  GuiInt    current_;
  GuiString execmode_;
  GuiInt    delay_;
  GuiInt    inc_amount_;
  GuiInt    send_amount_;
  int       inc_;
  bool      loop_;
  bool      use_row_;
  bool      didrun_;

  GuiString guifilename_;
  TimeDataFile datafile_;
  
  void send_selection(int which, int amount);
  int increment(int which, int lower, int upper);
  
public:
  TimeDataReader(GuiContext*);
  virtual ~TimeDataReader();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
};



DECLARE_MAKER(TimeDataReader)
TimeDataReader::TimeDataReader(GuiContext* ctx)
  : Module("TimeDataReader", ctx, Filter, "DataIO", "CardioWave"),
    row_or_col_(ctx->subVar("row_or_col")),
    selectable_min_(ctx->subVar("selectable_min")),
    selectable_max_(ctx->subVar("selectable_max")),
    selectable_inc_(ctx->subVar("selectable_inc")),
    selectable_units_(ctx->subVar("selectable_units")),
    range_min_(ctx->subVar("range_min")),
    range_max_(ctx->subVar("range_max")),
    playmode_(ctx->subVar("playmode")),
    dependence_(ctx->subVar("dependence")),
    current_(ctx->subVar("current")),
    execmode_(ctx->subVar("execmode")),
    delay_(ctx->subVar("delay")),
    inc_amount_(ctx->subVar("inc-amount")),
    send_amount_(ctx->subVar("send-amount")),
    guifilename_(ctx->subVar("filename")), 
    inc_(1),
    loop_(false),
    use_row_(false),
    didrun_(false)
{
}

void TimeDataReader::send_selection(int which, int amount)
{

  SCIRun::MatrixOPort *ovec = (SCIRun::MatrixOPort *)get_oport("Vector");
  if (!ovec) {
    error("Unable to initialize output port 'Vector'.");
    return;
  }

  SCIRun::NrrdOPort *onrrdvec = (SCIRun::NrrdOPort *)get_oport("NrrdVector");
  if (!onrrdvec) {
    error("Unable to initialize output port 'NrrdVector'.");
    return;
  }


  SCIRun::MatrixOPort *osel = (SCIRun::MatrixOPort *)get_oport("Selected Index");
  if (!osel) {
    error("Unable to initialize oport 'Selected Index'.");
    return;
  }
  
  if (ovec->nconnections())
  {
    SCIRun::MatrixHandle matrix(0);
    if (use_row_) 
    {
      try
      {
        datafile_.getrowmatrix(matrix,which,which+(amount-1));
      }
      catch (TimeDataFileException e)
      {
        error(e.message());
        return;
      }
    }
    else
    {
      try
      {
        datafile_.getcolmatrix(matrix,which,which+(amount-1));  
      }
      catch (TimeDataFileException e)
      {
        error(e.message());
        return;
      }

    }
    ovec->send(matrix);
  }
  
  if (onrrdvec->nconnections())
  {
    SCIRun::NrrdDataHandle nrrd(0);
    if (use_row_) 
    {
      try
      {
        datafile_.getrownrrd(nrrd,which,which+(amount-1));
      }
      catch (TimeDataFileException e)
      {
        error(e.message());
        return;
      }

    }
    else
    {
      try
      {
        datafile_.getcolnrrd(nrrd,which,which+(amount-1));  
      }
      catch (TimeDataFileException e)
      {
        error(e.message());
        return;
      }

    }
    onrrdvec->send(nrrd);
  }
  
  
  SCIRun::ColumnMatrix *selected = scinew SCIRun::ColumnMatrix(1);
  selected->put(0, 0, (double)which);
  osel->send(MatrixHandle(selected));
}


int TimeDataReader::increment(int which, int lower, int upper)
{
  // Do nothing if no range.
  if (upper == lower) {
    if (playmode_.get() == "once")
      execmode_.set( "stop" );
    return upper;
  }
  const int inc_amount = Max(1, Min(upper, inc_amount_.get()));

  which += inc_ * inc_amount;

  if (which > upper) {
    if (playmode_.get() == "bounce1") {
      inc_ *= -1;
      return increment(upper, lower, upper);
    } else if (playmode_.get() == "bounce2") {
      inc_ *= -1;
      return upper;
    } else {
      if (playmode_.get() == "once")
	execmode_.set( "stop" );
      return lower;
    }
  }
  if (which < lower) {
    if (playmode_.get() == "bounce1") {
      inc_ *= -1;
      return increment(lower, lower, upper);
    } else if (playmode_.get() == "bounce2") {
      inc_ *= -1;
      return lower;
    } else {
      if (playmode_.get() == "once")
	execmode_.set( "stop" );
      return upper;
    }
  }
  return which;
}


TimeDataReader::~TimeDataReader()
{
}

void TimeDataReader::execute()
{
  update_state(NeedData);
  
  std::string filename;
  
  NrrdIPort *filenameport;
  if ((filenameport = static_cast<NrrdIPort *>(getIPort("Filename"))))
  {
    NrrdDataHandle nrrdH;
    if (filenameport->get(nrrdH))
    {
        NrrdString fname(nrrdH);
        std::string filename = fname.getstring();
        guifilename_.set(filename);
        ctx->reset();
    }
  }  
  filename = guifilename_.get();

  NrrdOPort *filenameoport;
  if ((filenameoport = static_cast<NrrdOPort *>(getOPort("Filename"))))
  {
      NrrdString ns(filename);
      NrrdDataHandle nrrdH = ns.gethandle(); 
      filenameoport->send(nrrdH);
  }
  
  ctx->reset();
  
  try
  {
    datafile_.open(filename);
  }
  catch (TimeDataFileException e)
  {
    error(std::string("Could not open header file: ") + filename + "(" + e.message() + ")" );
    return;
  }

  update_state(JustStarted);
  
  bool changed_p = true;
  
  if (didrun_ == false) { changed_p = true; }

  bool use_row = (row_or_col_.get() == "row");
  if( use_row_ != use_row ) {
    changed_p = true;
  }

  if (use_row_)
  {
    double minlabel = 0.0;
    if (minlabel != selectable_min_.get()) 
    {
        selectable_min_.set(minlabel);
        changed_p = true;
    }

    double maxlabel = static_cast<double>(datafile_.getnrows()) - 1.0;
    if (maxlabel != selectable_max_.get()) 
    {
        selectable_max_.set(maxlabel);
        changed_p = true;
    }

    int increments = datafile_.getnrows();
    if (increments != selectable_inc_.get())
    {
        selectable_inc_.set(increments);
        changed_p = true;
    }
  }
  else
  {
    double minlabel = 0.0;
    if (minlabel != selectable_min_.get()) 
    {
        selectable_min_.set(minlabel);
        changed_p = true;
    }

    double maxlabel = static_cast<double>(datafile_.getncols()) - 1.0;
    if (maxlabel != selectable_max_.get()) 
    {
        selectable_max_.set(maxlabel);
        changed_p = true;
    }

    int increments = datafile_.getncols();
    if (increments != selectable_inc_.get())
    {
        selectable_inc_.set(increments);
        changed_p = true;
    }   
  }
    
  if (changed_p) gui->execute(id + " update_range");
  reset_vars();

  int which;
  
  // If there is a current index matrix, use it.
  SCIRun::MatrixIPort *icur = (SCIRun::MatrixIPort *)get_iport("Current Index");
  if (!icur) {
    error("Unable to initialize iport 'Current Index'.");
    return;
  }

  int send_amount;
  if (use_row_)
  {
    send_amount = Max(1, Min(datafile_.getnrows(), send_amount_.get()));
  }
  else
  {
    send_amount = Max(1, Min(datafile_.getncols(), send_amount_.get()));  
  }

  
  SCIRun::MatrixHandle currentH;
  if (icur->get(currentH) && currentH.get_rep()) 
  {
    use_row_ = use_row;
    which = (int)(currentH->get(0, 0));
    send_selection(which, send_amount);
  } 
  else 
  {
    // Get the current start and end.
    ctx->reset();
    const int start = range_min_.get();
    const int end = range_max_.get();

    int lower = start;
    int upper = end;
    if (lower > upper) {int tmp = lower; lower = upper; upper = tmp; }

    // Update the increment.
    if (changed_p || playmode_.get() == "once" || playmode_.get() == "loop")
      inc_ = (start>end)?-1:1;

    // If the current value is invalid, reset it to the start.
    if (current_.get() < lower || current_.get() > upper) 
    {
      current_.set(start);
      inc_ = (start>end)?-1:1;
    }

    // Cash execmode and reset it in case we bail out early.
    const string execmode = execmode_.get();

    which = current_.get();

    // If updating, we're done for now.
    if ((didrun_ == false)||(use_row!=use_row_)) 
    {
      use_row_ = use_row;
      send_selection(which, send_amount);
      didrun_ = true;
    } 
    else if (execmode == "update") 
    {
    } 
    else if (execmode == "step") 
    {
      which = increment(current_.get(), lower, upper);
      send_selection(which, send_amount);

    } 
    else if (execmode == "stepb") 
    {
      inc_ *= -1;
      which = increment(current_.get(), lower, upper);
      inc_ *= -1;
      send_selection(which, send_amount);

    } 
    else if (execmode == "play") 
    {
      if( !loop_ ) 
      {
        if (playmode_.get() == "once" && which >= end) which = start;
      }

      send_selection(which, send_amount);

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
    
        int next = increment(which, lower, upper);    

        // Incrementing may cause a stop in the execmode so recheck.
        execmode_.reset();
        if( loop_ = (execmode_.get() == "play") ) 
        {
          which = next;

          want_to_execute();
        }
      }
    } 
    else 
    {
      if( execmode == "rewind" ) which = start;
      else if( execmode == "fforward" )	which = end;
    
      send_selection(which, send_amount);
    
      if (playmode_.get() == "inc_w_exec") 
      {
        which = increment(which, lower, upper);
      }
    }
  }
  current_.set(which);
}

void
 TimeDataReader::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("TimeDataReader needs a minor command");
    return;
  } else Module::tcl_command(args, userdata);
}

} // End namespace CardioWave


