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
//    File   : EinthovenLeads.cc
//    Author : Martin Cole
//    Date   : Mon Mar  7 09:33:56 2005

#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/NrrdData.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <VS/Dataflow/Modules/Render/EinthovenLeads.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomDL.h>

namespace VS {
using namespace SCIRun;
using std::cerr;
using std::endl;


class EinthovenLeads : public Module
{
public:
  EinthovenLeads(GuiContext* ctx);
  virtual ~EinthovenLeads();
  virtual void		execute();
  virtual void		tcl_command(GuiArgs& args, void*);
  
private:
  GuiInt                               gui_lead_I_;
  GuiInt                               gui_lead_II_;
  GuiInt                               gui_lead_III_;
  
  int                                  last_gen_;
  int                                  count_;
  NrrdDataHandle                       nrrd_out_;
  int                                  text_id_;
};



DECLARE_MAKER(EinthovenLeads)

EinthovenLeads::EinthovenLeads(GuiContext* ctx) :
  Module("EinthovenLeads", ctx, Filter, "Render", "VS"),
  gui_lead_I_(ctx->subVar("lead_I")),
  gui_lead_II_(ctx->subVar("lead_II")),
  gui_lead_III_(ctx->subVar("lead_III")),
  last_gen_(-1),
  count_(0),
  nrrd_out_(0),
  text_id_(-1)
{
}

EinthovenLeads::~EinthovenLeads()
{
}

void
EinthovenLeads::execute()
{
  FieldIPort *ifport = (FieldIPort*)get_iport("Torso");
  
  if (! ifport) 
  {
    error("Unable to initialize input port Torso.");
    return;
  }

  FieldHandle torso;
  ifport->get(torso);

  if (! torso.get_rep())
  {
    error ("Unable to get input data.");
    return;
  } 

  vector<Point> pos(3);
  if (last_gen_ != torso->generation) {
    last_gen_ = torso->generation;
    NrrdDataHandle ndata;
    const TypeDescription *torso_td = torso->get_type_description();
    CompileInfoHandle ci = EinthovenLeadsAlgo::get_compile_info(torso_td);
    Handle<EinthovenLeadsAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;
    vector<double> values(3, 0.0);
    if (algo->get_values(torso, gui_lead_I_.get(), gui_lead_II_.get(), 
			 gui_lead_III_.get(), values, pos)) 
    {
      if (count_ % 100 == 0) {
	int sz = (count_ / 100 + 1) * 100;
	ndata = new NrrdData();

	ndata->nrrd->axis[0].kind = nrrdKindDomain;
	ndata->nrrd->axis[1].kind = nrrdKindDomain;
	nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			nrrdCenterNode);
	ndata->nrrd->axis[0].label = strdup("leads");
	ndata->nrrd->axis[1].label = strdup("time");
	nrrdAlloc(ndata->nrrd, nrrdTypeFloat, 2, 3, sz);
	memset(ndata->nrrd->data, 0, count_ * 3 * sizeof(float));
	if (nrrd_out_.get_rep()) {
	  memcpy(ndata->nrrd->data, nrrd_out_->nrrd->data, 
		 count_ * 3 * sizeof(float));
	  
	}
	nrrd_out_ = ndata;
      }
      float last[3];

      float *dat = (float*)nrrd_out_->nrrd->data;
      
      if (count_) {
	last[0] = dat[3 * (count_ - 1)];
	last[1] = dat[3 * (count_ - 1) + 1];
	last[2] = dat[3 * (count_ - 1) + 2];
	dat[3 * count_] = (values[0] - values[1])*0.7 + last[0]*0.3;
	dat[3 * count_ + 1] = (values[2] - values[1])*0.7 + last[1]*0.3;
	dat[3 * count_ + 2] = (values[2] - values[0])*0.7 + last[2]*0.3;
      
      } else {
	dat[3 * count_] = values[0] - values[1];
	dat[3 * count_ + 1] = values[2] - values[1];
	dat[3 * count_ + 2] = values[2] - values[0];
      }
      ++count_;      
      // build the nrrd output.
    } else {
      error("Invalid Input Field, possibly wrong or no data");
    }
  }
  NrrdOPort *oport = (NrrdOPort*)get_oport("Values");
  if (! oport) 
  {
    error("Unable to initialize output port Values.");
    return;
  }  
  
  GeometryOPort *goport = (GeometryOPort *)get_oport("Electrodes");
  if (text_id_ != -1) goport->delObj(text_id_);

  GeomTexts *texts = scinew GeomTexts();
  GeomHandle text_switch = scinew GeomSwitch(scinew GeomDL(texts));
  
  Color c(.5, .95, .95);
  texts->add(string("LA"), pos[0], c);
  texts->add(string("RA"), pos[1], c);
  texts->add(string("LF"), pos[2], c);
  texts->set_font_index(18);

  text_id_ = goport->addObj(text_switch, "Electrode Location");

  oport->send(nrrd_out_);
  
}

void
EinthovenLeads::tcl_command(GuiArgs& args, void* userdata) 
{
  if(args.count() < 2) {
    args.error("EinthovenLeads needs a minor command");
    return;
  } else if(args[1] == "reset") {
    nrrd_out_ = 0;
  } else {
    Module::tcl_command(args, userdata);
  }
}

template <>
bool
to_double(const Vector&, double&)
{
  return false;
}

template <>
bool
to_double(const Tensor &, double &)
{
  return false;
}

CompileInfoHandle
EinthovenLeadsAlgo::get_compile_info(const TypeDescription *td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("EinthovenLeadsAlgoT");
  static const string base_class_name("EinthovenLeadsAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("VS");
  td->fill_compile_info(rval);
  return rval;
}



} // End namespace VS
