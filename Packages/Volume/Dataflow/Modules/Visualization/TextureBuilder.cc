/*
 *  Texture.cc:
 *
 *  Written by:
 *   kuzimmer
 *   TODAY'S DATE HERE
 *
 */

#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/share/share.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Packages/Volume/Core/Util/Utils.h>
#include <Packages/Volume/Core/Datatypes/Texture.h>
#include <Packages/Volume/Dataflow/Ports/TexturePort.h>
#include <Packages/Volume/Core/Algorithms/TextureBuilderAlgo.h>

namespace Volume {

using namespace SCIRun;

class PSECORESHARE TextureBuilder : public Module {
public:
  TextureBuilder(GuiContext*);

  virtual ~TextureBuilder();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiDouble gui_vminval_;
  GuiDouble gui_vmaxval_;
  GuiDouble gui_gminval_;
  GuiDouble gui_gmaxval_;
  
  GuiInt gui_fixed_;
  GuiInt gui_card_mem_;

  int vfield_last_generation_;
  int gfield_last_generation_;
  double vminval_, vmaxval_;
  double gminval_, gmaxval_;
  
  ProgressReporter my_reporter_;
  template<class Reporter> bool build_texture(Reporter*, FieldHandle, FieldHandle);
  template<class Reporter> bool replace_texture(Reporter*, FieldHandle, FieldHandle);

  bool new_vfield(FieldHandle field);
  bool new_gfield(FieldHandle field);

  TextureHandle texture_;
};


DECLARE_MAKER(TextureBuilder)
  
TextureBuilder::TextureBuilder(GuiContext* ctx)
  : Module("TextureBuilder", ctx, Source, "Visualization", "Volume"),
    gui_vminval_(ctx->subVar("vmin")),
    gui_vmaxval_(ctx->subVar("vmax")),
    gui_gminval_(ctx->subVar("gmin")),
    gui_gmaxval_(ctx->subVar("gmax")),
    gui_fixed_(ctx->subVar("is_fixed")),
    gui_card_mem_(ctx->subVar("card_mem")),
    vfield_last_generation_(-1), gfield_last_generation_(-1),
    texture_(0) {
}

TextureBuilder::~TextureBuilder() {
}

void
TextureBuilder::execute()
{
  FieldIPort* ivfield = (FieldIPort *)get_iport("Scalar Field");
  FieldIPort* igfield = (FieldIPort*)get_iport("Gradient Field");
  TextureOPort* otexture = (TextureOPort *)get_oport("Texture");
  
  if(!ivfield) {
    error("Unable to initialize input ports.");
    return;
  }

  FieldHandle vfield;
  ivfield->get(vfield);
  if(!vfield.get_rep()) {
    error("Field has no representation.");
    return;
  }
  
  if (vfield->generation != vfield_last_generation_) {
    // new field
    if (!new_vfield(vfield)) return;
    vfield_last_generation_ = vfield->generation;
  }

  FieldHandle gfield;
  if(igfield) {
    igfield->get(gfield);
    if(gfield.get_rep()) {
      if (gfield->generation != gfield_last_generation_) {
        // new field
        if (!new_gfield(gfield)) return;
        gfield_last_generation_ = gfield->generation;
      }
    }
  }

  if(build_texture(&my_reporter_, vfield, gfield)) {
    otexture->send(texture_);
  }
}

template<class Reporter>
bool
TextureBuilder::build_texture(Reporter* reporter, FieldHandle vfield, FieldHandle gfield)
{
  // start new algorithm based code
  const TypeDescription* td = vfield->get_type_description();
  cerr << "DC: type description = " << td->get_name() << endl;
  LockingHandle<TextureBuilderAlgoBase> builder;
  CompileInfoHandle ci = TextureBuilderAlgoBase::get_compile_info(td);
  if (!DynamicCompilation::compile(ci, builder, reporter)) {
    reporter->error("Texture Builder can not work on this Field");
    return false;
  }
  texture_ = TextureHandle(builder->build(vfield, gfield,
                                          gui_card_mem_.get(), 
                                          vminval_, vmaxval_, gminval_, gmaxval_));
  return true;
}


template<class Reporter>
bool
TextureBuilder::replace_texture(Reporter *reporter,
                                FieldHandle vfield, FieldHandle gfield)
{
  return false;
}

void
TextureBuilder::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

bool
TextureBuilder::new_vfield(FieldHandle vfield)
{
  const string type = vfield->get_type_description()->get_name();

  ScalarFieldInterfaceHandle sfi = vfield->query_scalar_interface(this);
  if (!sfi.get_rep()) {
    error("Input field does not contain scalar data.");
    return false;
  }

  // set vmin/vmax
  pair<double, double> vminmax;
  sfi->compute_min_max(vminmax.first, vminmax.second);
  if(vminmax.first != vminval_ || vminmax.second != vmaxval_) {
    if(!gui_fixed_.get()) {
      gui_vminval_.set(vminmax.first);
      gui_vmaxval_.set(vminmax.second);
    }
    vminval_ = vminmax.first;
    vmaxval_ = vminmax.second;
  }

  return true;
}

  
bool
TextureBuilder::new_gfield(FieldHandle gfield)
{
  // set gmin/gmax
  LatVolField<Vector>* gfld = 
    dynamic_cast<LatVolField<Vector>*>(gfield.get_rep());

  if (gfld) {
    
    FData3d<Vector>::const_iterator bi, ei;
    bi = gfld->fdata().begin();
    ei = gfld->fdata().end();

    double gminval = std::numeric_limits<double>::max();
    double gmaxval = -gminval;
    while (bi != ei)
    {
      Vector v = *bi;
      double g = v.length();
      if (g < gminval) gminval = g;
      if (g > gmaxval) gmaxval = g;
      ++bi;
    }

    if(!gui_fixed_.get()) {
      gui_gminval_.set(gminval);
      gui_gmaxval_.set(gmaxval);
    }
    
    gminval_ = gminval;
    gmaxval_ = gmaxval;
  }  

  return true;
}

} // End namespace Volume


