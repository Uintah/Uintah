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

#include <Packages/Teem/Dataflow/Ports/NrrdPort.h>

#include <Packages/Volume/Core/Util/Utils.h>
#include <Packages/Volume/Core/Util/VideoCardInfo.h>
#include <Packages/Volume/Core/Datatypes/Texture.h>
#include <Packages/Volume/Dataflow/Ports/TexturePort.h>
#include <Packages/Volume/Core/Algorithms/NrrdTextureBuilderAlgo.h>

#include <sstream>
using std::ostringstream;

namespace Volume {

using namespace SCIRun;
using namespace SCITeem;

class PSECORESHARE NrrdTextureBuilder : public Module {
public:
  NrrdTextureBuilder(GuiContext*);

  virtual ~NrrdTextureBuilder();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiInt gui_card_mem_;
  GuiInt gui_card_mem_auto_;
  int card_mem_;
  
  int nvfield_last_generation_;
  int gmfield_last_generation_;
  
  ProgressReporter my_reporter_;
  template<class Reporter> bool build_texture(Reporter*, NrrdDataHandle, NrrdDataHandle);
  template<class Reporter> bool replace_texture(Reporter*, NrrdDataHandle, NrrdDataHandle);
  NrrdTextureBuilderAlgo builder_;
  TextureHandle texture_;
};


DECLARE_MAKER(NrrdTextureBuilder)
  
NrrdTextureBuilder::NrrdTextureBuilder(GuiContext* ctx)
  : Module("NrrdTextureBuilder", ctx, Source, "Visualization", "Volume"),
    gui_card_mem_(ctx->subVar("card_mem")),
    gui_card_mem_auto_(ctx->subVar("card_mem_auto")),
    nvfield_last_generation_(-1), gmfield_last_generation_(-1),
    texture_(0)
{
  card_mem_ = video_card_memory_size();
}

NrrdTextureBuilder::~NrrdTextureBuilder() {
}

void
NrrdTextureBuilder::execute()
{
  if(card_mem_ != 0 && gui_card_mem_auto_.get()) {
    ostringstream cmd;
    cmd << id << " set_card_mem " << card_mem_ << "; " <<  id << " state";
    gui->execute(cmd.str().c_str());
  } else if(card_mem_ == 0) {
    ostringstream cmd;
    cmd << id << " set_card_mem_auto 0; " << id << " state";
    gui->execute(cmd.str().c_str());
  }

  NrrdIPort* ivfield = (NrrdIPort*)get_iport("Value / Normal-Value Nrrd");
  NrrdIPort* igfield = (NrrdIPort*)get_iport("Gradmag Nrrd");
  TextureOPort* otexture = (TextureOPort *)get_oport("Texture");

  if(!ivfield) {
    error("Unable to initialize input ports.");
    return;
  }

  // check rep
  NrrdDataHandle nvfield;
  ivfield->get(nvfield);
  if(!nvfield.get_rep()) {
    error("Field has no representation.");
    return;
  }

  // check type
  if (nvfield->nrrd->type != nrrdTypeUChar) {
    error("Input nrrd type is not unsigned char");
    return;
  }
  
  if (nvfield->generation != nvfield_last_generation_) {
    nvfield_last_generation_ = nvfield->generation;
  }

  NrrdDataHandle gmfield;
  if(igfield) {
    igfield->get(gmfield);
    if(gmfield.get_rep()) {
      if (gmfield->generation != gmfield_last_generation_) {
        gmfield_last_generation_ = gmfield->generation;
      }
      // check type
      if (gmfield->nrrd->type != nrrdTypeUChar) {
        error("Input nrrd type is not unsigned char");
        return;
      }
    }
  }

  if(build_texture(&my_reporter_, nvfield, gmfield)) {
    otexture->send(texture_);
  }
}

template<class Reporter>
bool
NrrdTextureBuilder::build_texture(Reporter* reporter,
                                  NrrdDataHandle nvfield, NrrdDataHandle gmfield)
{
  texture_ = TextureHandle(builder_.build(nvfield, gmfield, gui_card_mem_.get()));
  return true;
}


template<class Reporter>
bool
NrrdTextureBuilder::replace_texture(Reporter *reporter,
                                    NrrdDataHandle nvfield, NrrdDataHandle gmfield)
{
  return false;
}

void
NrrdTextureBuilder::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Volume
