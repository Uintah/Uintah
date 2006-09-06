#if !defined(TEXTURE_BUILDER_H)
#define TEXTURE_BUILDER_H

#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/GuiInterface/GuiContext.h>
#include <Core/Volume/Texture.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {

class TextureBuilder : public Module
{
public:
  TextureBuilder(GuiContext* ctx, const std::string& name="TextureBuilder",
                 SchedClass sc = Source,  const string& cat="Visualization", 
                 const string& pack="SCIRun");
  virtual ~TextureBuilder();

  virtual void execute();

protected:
  TextureHandle tHandle_;

  GuiDouble gui_vminval_;
  GuiDouble gui_vmaxval_;
  GuiDouble gui_gminval_;
  GuiDouble gui_gmaxval_;

  GuiInt gui_fixed_;
  GuiInt gui_card_mem_;
  GuiInt gui_card_mem_auto_;
  int card_mem_;

  int vfield_last_generation_;
  int gfield_last_generation_;
  double vminval_, vmaxval_;
  double gminval_, gmaxval_;
};

} // end namespace SCIRun

#endif
