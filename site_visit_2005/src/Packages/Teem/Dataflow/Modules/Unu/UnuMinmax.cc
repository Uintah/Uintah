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
 *  UnuMinmax.cc:  Print out min and max values in one or more nrrds. Unlike other
 *  modules, this doesn't produce a nrrd. It only prints to the UI the max values 
 *  found in the input nrrd(s), and it also indicates if there are non-existant values.
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/StringUtil.h>

#include <Dataflow/Ports/NrrdPort.h>


namespace SCITeem {

using namespace SCIRun;

class UnuMinmax : public Module {
public:
  UnuMinmax(GuiContext*);

  virtual ~UnuMinmax();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  NrrdOPort*         onrrd_;

  NrrdDataHandle     onrrd_handle_;  //! the cached output nrrd handle.
  vector<int>        in_generation_; //! all input generation nums.
                                    //! target type for output nrrd.
  GuiInt             nrrds_;
  vector<GuiDouble*> mins_;
  vector<GuiDouble*> maxs_;
};


DECLARE_MAKER(UnuMinmax)
UnuMinmax::UnuMinmax(GuiContext* ctx)
  : Module("UnuMinmax", ctx, Source, "UnuAtoM", "Teem"),
    onrrd_(0), onrrd_handle_(0), in_generation_(0),
    nrrds_(ctx->subVar("nrrds"))
{
}

UnuMinmax::~UnuMinmax(){
}

void
 UnuMinmax::execute(){
  port_range_type range = get_iports("Nrrds");
  if (range.first == range.second) { return; }

  unsigned int i = 0;
  vector<NrrdDataHandle> nrrds;
  bool do_join = false;
  port_map_type::iterator pi = range.first;
  int num_nrrds = 0;
  while (pi != range.second)
  {
    NrrdIPort *inrrd = (NrrdIPort *)get_iport(pi->second);
    NrrdDataHandle nrrd;
    
    if (inrrd->get(nrrd) && nrrd.get_rep()) {
      // check to see if we need to do the join or can output the cached onrrd.
      if (in_generation_.size() <= i) {
	// this is a new input, never been joined.
	do_join = true;
	in_generation_.push_back(nrrd->generation);
      } else if (in_generation_[i] != nrrd->generation) {
	// different input than last execution
	do_join = true;
	in_generation_[i] = nrrd->generation;
      }

      nrrds.push_back(nrrd);
      num_nrrds++;
    }
    ++pi; ++i;
  }

  if (num_nrrds != nrrds_.get()) {
    do_join = true;
  }

  nrrds_.set(num_nrrds);

  vector<Nrrd*> arr(nrrds.size());

  if (do_join) {
    vector<double> mins, maxs;

    int i = 0;
    vector<NrrdDataHandle>::iterator iter = nrrds.begin();
    while(iter != nrrds.end()) {
      NrrdDataHandle nh = *iter;
      ++iter;

      NrrdData* cur_nrrd = nh.get_rep();
      NrrdRange *range = nrrdRangeNewSet(cur_nrrd->nrrd, nrrdBlind8BitRangeFalse);
      mins.push_back(range->min);
      maxs.push_back(range->max);
      ++i;
    }
    

    // build list string
    for (int i=0; i<(int)mins.size(); i++) {
      ostringstream min_str, max_str;
      min_str << "min" << i;
      if ((int)mins_.size() <= i)
	mins_.push_back(new GuiDouble(ctx->subVar(min_str.str())));
      max_str << "max" << i;
      if ((int)maxs_.size() <= i)
	maxs_.push_back(new GuiDouble(ctx->subVar(max_str.str())));
    }
    gui->execute(id + " init_axes");

    for (int i=0; i<(int)mins.size(); i++) {
      mins_[i]->set(mins[i]);
      mins_[i]->reset();
      maxs_[i]->set(maxs[i]);
      maxs_[i]->reset();
    }
    gui->execute(id + " make_min_max");

  }
}

void
 UnuMinmax::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


