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

  GuiDouble gui_minval_;
  GuiDouble gui_maxval_;
  GuiInt gui_fixed_;
  GuiInt gui_card_mem_;

  int last_generation_;
  double minval_, maxval_;

  ProgressReporter my_reporter_;
  template<class Reporter> bool build_texture( Reporter *, FieldHandle);
  template<class Reporter> bool replace_texture( Reporter *, FieldHandle);


  bool new_field(FieldHandle field);
  TextureHandle texture_;
};


DECLARE_MAKER(TextureBuilder)
TextureBuilder::TextureBuilder(GuiContext* ctx)
  : Module("TextureBuilder", ctx, Source, "Visualization", "Volume"),
    gui_minval_(ctx->subVar("min")),
    gui_maxval_(ctx->subVar("max")),
    gui_fixed_(ctx->subVar("is_fixed")),
    gui_card_mem_(ctx->subVar("card_mem")),
    last_generation_(-1),
    texture_(0)
{
}

TextureBuilder::~TextureBuilder(){
}

void
 TextureBuilder::execute()
{
  FieldIPort *infield = (FieldIPort *)get_iport("Scalar Field");
  TextureOPort *otexture = (TextureOPort *)get_oport("Texture");

  FieldHandle field;

  if(!infield){
    error("Unable to initialize iport 'Field'.");
    return;
  }

  infield->get(field);
  if(!field.get_rep())
  {
    error("Field has no representation.");
    return;
  }
  

  if (field->generation != last_generation_){
    // new field
    if (!new_field( field )) return;
    last_generation_ = field->generation;
  }


  if( build_texture( &my_reporter_, field ) ){
    otexture->send( texture_ );
  }
}

template<class Reporter>
bool
TextureBuilder::build_texture( Reporter *reporter, FieldHandle texfld)
{
  // start new algorithm based code
  const TypeDescription *td = texfld->get_type_description();
  cerr << "DC: type description = " << td->get_name() << endl;
  LockingHandle<TextureBuilderAlgoBase> builder;
  CompileInfoHandle ci = TextureBuilderAlgoBase::get_compile_info(td);
  if ( !DynamicCompilation::compile(ci, builder, reporter ) ) {
    reporter->error("Texture Builder can not work on this Field");
    return false;
  }

  texture_ = TextureHandle( builder->build( texfld, gui_card_mem_.get(), 
					    minval_, maxval_ ) );

  return true;
}
template<class Reporter>
bool
TextureBuilder::replace_texture( Reporter *reporter, FieldHandle texfld)
{
  // start new algorithm based code
  const TypeDescription *td = texfld->get_type_description();
  cerr << "DC: type description = " << td->get_name() << endl;
  LockingHandle<TextureBuilderAlgoBase> builder;
  CompileInfoHandle ci = TextureBuilderAlgoBase::get_compile_info(td);
  if ( !DynamicCompilation::compile(ci, builder,reporter) ) {
    reporter->error("Texture Builder can not work on this Field");
    return false;
  }

  return false;
}
void
 TextureBuilder::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


bool
TextureBuilder::new_field(FieldHandle field)
{
  const string type = field->get_type_description()->get_name();

  ScalarFieldInterfaceHandle sfi = field->query_scalar_interface(this);
  if (!sfi.get_rep())
  {
    error("Input field does not contain scalar data.");
    return false;
  }
  // Set min/max
  pair<double, double> minmax;
  sfi->compute_min_max(minmax.first, minmax.second);
  if(minmax.first != minval_ || minmax.second != maxval_){
    if( !gui_fixed_.get()) {
      gui_minval_.set( minmax.first );
      gui_maxval_.set( minmax.second );
    }
    minval_ = minmax.first;
    maxval_ = minmax.second;
  }
  return true;
}

} // End namespace Volume


