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
 *  UnuAxinfo.cc:
 *
 *  Written by:
 *   Darby Van Uitert
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

namespace SCITeem {
using namespace SCIRun;

class UnuAxinfo : public Module {
public:
  UnuAxinfo(GuiContext* ctx);
  virtual ~UnuAxinfo();

  virtual void execute();

  GuiInt              axis_;
  GuiString           label_;
  GuiString           kind_;
  GuiDouble           min_;
  GuiDouble           max_;
  GuiDouble           spacing_;
  GuiString           space_dir_;
  GuiInt              use_label_;
  GuiInt              use_kind_;
  GuiInt              use_min_;
  GuiInt              use_max_;
  GuiInt              use_spacing_;
  GuiInt              use_space_dir_;
private:
  int                 generation_;
};

DECLARE_MAKER(UnuAxinfo)

UnuAxinfo::UnuAxinfo(GuiContext* ctx)
  : Module("UnuAxinfo", ctx, Source, "UnuAtoM", "Teem"),
    axis_(ctx->subVar("axis")),
    label_(ctx->subVar("label")),
    kind_(ctx->subVar("kind")),
    min_(ctx->subVar("min")),
    max_(ctx->subVar("max")),
    spacing_(ctx->subVar("spacing")),
    space_dir_(ctx->subVar("spaceDir")),
    use_label_(ctx->subVar("use_label")),
    use_kind_(ctx->subVar("use_kind")),
    use_min_(ctx->subVar("use_min")),
    use_max_(ctx->subVar("use_max")),
    use_spacing_(ctx->subVar("use_spacing")),
    use_space_dir_(ctx->subVar("use_spaceDir")),
    generation_(-1)
{
}

UnuAxinfo::~UnuAxinfo(){
}


void UnuAxinfo::execute()
{
  NrrdIPort *iport = (NrrdIPort*)get_iport("Nrrd"); 
  NrrdOPort *oport = (NrrdOPort*)get_oport("Nrrd");
  
  update_state(NeedData);
  
  // The input port (with data) is required.
  NrrdDataHandle nh;
  if (!iport->get(nh)) 
    {
      return;
    }
  
  int axis = axis_.get();
  if( !nh.get_rep() || generation_ != nh->generation ) {
    if (axis >= nh->nrrd->dim) {
      error("Please specify an axis within proper range.");
      return;
    }
    
    generation_ = nh->generation;
  }
  
  reset_vars();

  Nrrd *nin = nh->nrrd;
  Nrrd *nout = nrrdNew();
  
  // copy input nrrd and modify its label, kind, min, max and spacing
  if (nrrdCopy(nout, nin)) {
    char *err = biffGetDone(NRRD);
    error(string("Trouble copying input nrrd: ") +  err);
    msgStream_ << "  input Nrrd: nin->dim=" << nin->dim << "\n";
    free(err);
    return;
  }
  
  if (use_label_.get() && strlen(label_.get().c_str())) {
    nout->axis[axis].label = (char*)airFree(nout->axis[axis].label);
    nout->axis[axis].label = airStrdup(const_cast<char*>(label_.get().c_str()));
  }

  if (use_kind_.get()) { 
    string kind = kind_.get();
    if (kind == "nrrdKindDomain") {
      nout->axis[axis].kind = nrrdKindDomain;
    } else if (kind == "nrrdKindScalar") {
      nout->axis[axis].kind = nrrdKindScalar;
    } else if (kind == "nrrdKind3Color") {
      nout->axis[axis].kind = nrrdKind3Color;
    } else if (kind == "nrrdKind3Vector") {
      nout->axis[axis].kind = nrrdKind3Vector;
    } else if (kind == "nrrdKind3Normal") {
      nout->axis[axis].kind = nrrdKind3Normal;
    } else if (kind == "nrrdKind3DSymMatrix") {
      nout->axis[axis].kind = nrrdKind3DSymMatrix;
    } else if (kind == "nrrdKind3DMaskedSymMatrix") {
      nout->axis[axis].kind = nrrdKind3DMaskedSymMatrix;
    } else if (kind == "nrrdKind3DMatrix") {
      nout->axis[axis].kind = nrrdKind3DMatrix;
    } else if (kind == "nrrdKindList") {
      nout->axis[axis].kind = nrrdKindList;
    } else if (kind == "nrrdKindStub") {
      nout->axis[axis].kind = nrrdKindStub;
    } else {
      nout->axis[axis].kind = nrrdKindUnknown;
    }
  }

  // determine if this nrrd has min,max,spacing defined,
  // or spaceDirection vectors
  bool has_space = false;
  
  if (airExists(nout->axis[axis].spaceDirection[0]))
    has_space = true;


  // min, max and spacing can be modified as long as the 
  // axis does NOT have spaceDirection vectors set.
  if (use_min_.get() && airExists(min_.get())) { 
    if (has_space) 
      error("ERROR: Cannot change min point when spaceDirection vectors are defined.");
    else
      nout->axis[axis].min = min_.get();
  }
  
  if (use_max_.get() && airExists(max_.get())) {
    if (has_space) 
      error("ERROR: Cannot change max point when spaceDirection vectors are defined.");
    else
      nout->axis[axis].max = max_.get();
  }
  
  if (use_spacing_.get() && airExists(spacing_.get())) {
    if (has_space)
      error("ERROR: Cannot change spacing when spaceDirection vectors are defined.");
    else
      nout->axis[axis].spacing = spacing_.get();
  }
  
  string space_string = space_dir_.get();
  if (use_space_dir_.get() && space_string.length() > 0) {
    if (!has_space) 
      error("ERROR: Cannot change spaceDirection vector when min, max, or spacing is defined.");
    else {
      // parse the space string for values separated by commas (there
      // should be the same number of values as nout->spaceDim
      int i=0, start=0, end=0, which=0, counter=0;
      char ch;
      bool in_num = false;
      vector<double> spaceDirs;
      while (i < (int)space_string.length()) {
	ch = space_string[i];
	if(isspace(ch) || ch == ',') {
	  if (in_num) {
	    end = i;
	    spaceDirs.push_back(atof(space_string.substr(start, end-start).c_str()));
	    which++;
	    counter++;
	    in_num = false;
	  }
	} else if (i == (int)space_string.length()-1) {
	  if (!in_num) {
	    start = i;
	  }
	  end = i+1;
	  spaceDirs.push_back(atof(space_string.substr(start, end-start).c_str()));
	  which++;
	  counter++;
	  in_num = false;
	} else {
	  if(!in_num) {
	    start = i;
	    in_num = true;
	  }
	}	
	++i;
      }
      if ((int)spaceDirs.size() != (int)nout->spaceDim)
	error(string("ERROR: You must specify "+ to_string(nout->spaceDim) + " components for your spaceDirection vector."));
      else {
	for(int i=0; i<nout->spaceDim; i++)
	  nout->axis[axis].spaceDirection[i] = spaceDirs[i];
      }	
    }
  }
  
  if (!has_space) {
    double calc_max = nout->axis[axis].min + 
      (nout->axis[axis].spacing * (nout->axis[axis].size-1));
    
    if (nout->axis[axis].center == nrrdCenterCell) 
      calc_max = nout->axis[axis].min + 
	(nout->axis[axis].spacing * nout->axis[axis].size-1);
    
    if (airExists(nout->axis[axis].min) && 
	airExists(nout->axis[axis].max) &&
	airExists(nout->axis[axis].spacing) && 
	calc_max != nout->axis[axis].max) {
      remark("Warning: Output NRRD's min, max and spacing are not valid. Recalculating max.");
      nout->axis[axis].max = calc_max;
    }
  }

  // verify it is a valid nrrd going out
  if (nrrdCheck(nout))
    error("ERROR: Invalid output NRRD.");

  
  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  
  NrrdDataHandle out(nrrd);
  
  // Copy the properties.
  out->copy_properties(nh.get_rep());
  
  oport->send(out);

}

} // end SCITeem namespace

