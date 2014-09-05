/*
 *  Silhouettes.cc:
 *
 *  Written by:
 *   allen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

//#include <Dataflow/Widgets/ViewPointWidget.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Containers/Handle.h>

#include <Packages/PCS/Dataflow/Modules/Visualization/Silhouettes.h>

namespace PCS {

using namespace SCIRun;

extern GeometryData* SilhouettesGeometryData;

class Silhouettes : public Module {

public:
  Silhouettes(GuiContext*);

  virtual ~Silhouettes();

  virtual void widget_moved(bool last, BaseWidget*);

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

protected:
  GuiInt gui_build_field_;
  GuiInt gui_build_geom_;
  GuiDouble gui_color_r_;
  GuiDouble gui_color_g_;
  GuiDouble gui_color_b_;
  GuiInt gui_autoexec_;
  CrowdMonitor gui_widget_lock_;

//  ViewPointWidget* vpWidget_;

  int build_field_;
  int build_geom_;
  int auto_exec_;

  FieldHandle fHandle_;
  int geomID_;
  double isoval_;

  View view_;

  int fGeneration_;
  int cmGeneration_;

  bool error_;
};

DECLARE_MAKER(Silhouettes)
Silhouettes::Silhouettes(GuiContext* ctx)
  : Module("Silhouettes", ctx, Source, "Visualization", "PCS"),
    gui_build_field_(ctx->subVar("build_field")),
    gui_build_geom_(ctx->subVar("build_geom")),
    gui_color_r_(ctx->subVar("color-r")),
    gui_color_g_(ctx->subVar("color-g")),
    gui_color_b_(ctx->subVar("color-b")),
    gui_autoexec_(ctx->subVar("autoexecute")),
    gui_widget_lock_("Silhouettes widget lock"),
//    vpWidget_(0),
    build_field_(0),
    build_geom_(0),
    auto_exec_(0),
    fHandle_(0),
    geomID_(0),
    fGeneration_(-1),
    cmGeneration_(-1),
    error_( false )
{
}

Silhouettes::~Silhouettes(){
//  if (vpWidget_) delete vpWidget_;
}

void
Silhouettes::widget_moved(bool last, BaseWidget*)
{
  /*
  if (last) {
    if (vpWidget_) {
      gui_autoexec_.reset();

      if (gui_autoexec_.get())
	want_to_execute();
    }
  }
  */
}

void
Silhouettes::execute(){
  update_state(NeedData);
 
  bool update = false;


  FieldIPort* ifield_port = (FieldIPort *)get_iport("Input Field");

  FieldHandle fHandle;

  if (!ifield_port) {
    error( "Unable to initialize iport 'Input Field'.");
    return;
  }

  if (!(ifield_port->get(fHandle) && fHandle.get_rep())) {
    error( "No handle or representation in input field." );
    return;
  }

  if (!fHandle->query_scalar_interface(this).get_rep() ) {
    error( "This module only works on fields of scalar data.");
    return;
  }

  // Check to see if the input field has changed.
  if( fGeneration_ != fHandle->generation ) {

    fGeneration_ = fHandle->generation;

    update = true;
  }


  ColorMapIPort *icmap_port = (ColorMapIPort *)get_iport("Optional Color Map");

  if (!icmap_port) {
    error("Unable to initialize iport 'Optional Color Map'.");
    return;
  }

  ColorMapHandle cmHandle;
  bool have_ColorMap = false;
  if (icmap_port->get(cmHandle)) {
    if(!cmHandle.get_rep()) {
      error( "No colormap representation." );
      return;
    }   
     
    have_ColorMap = true;
    if( cmGeneration_ != cmHandle->generation ) {
    
      cmGeneration_ = cmHandle->generation;
      update = true;
    }
  }
  
  // Get a handle to the output geometry port.
  GeometryOPort* ogeom_port = (GeometryOPort *) get_oport("Silouette Geom");
  if (!ogeom_port) {
    error( "Unable to initialize oport 'Silhouette Geom'.");
    return;
  }

  if( ogeom_port->getNViewers() == 0 ) {
    error("Geometery port is not attached to a viewer.");
    return;
  }

  // Get the current view.
  GeometryData *geometry = ogeom_port->getData( 0, 0, GEOM_VIEW );

  if( geometry == 0 ) {
    error("Geometery port must be attached to the viewer.");
    return;    
  }

  /*
  if (!vpWidget_) {
    vpWidget_ = scinew ViewPointWidget(this, &gui_widget_lock_);
    vpWidget_->Connect(oview_port);

    GeomHandle widget = vpWidget_->GetWidget();
    oview_port->addObj(widget, "Silhouettes View Point", &gui_widget_lock_);
    oview_port->flushViews();
  }
  */

  int build_field = gui_build_field_.get();
  int build_geom  = gui_build_geom_.get();

  // If no data or a change recalcute.
  if( (build_field  && !fHandle_.get_rep()) ||
      (build_geom   && !geomID_ == -1     ) ||
      view_ != *(geometry->view) ||
      update ||
      error_ ) {

    update_state(JustStarted);

    error_ = false;

    build_field_ = build_field;
    build_geom_  = build_geom;
    
    // Stop showing the previous geometry.
    bool geomflush = false;

    if ( geomID_ ) {
      ogeom_port->delObj( geomID_ );
      geomID_ = 0;
      geomflush = true;
    }

    const TypeDescription *ftd = fHandle->get_type_description(0);
    const TypeDescription *ttd = fHandle->get_type_description(1);

    CompileInfoHandle ci = SilhouettesAlgo::get_compile_info(ftd, ttd);
    Handle<SilhouettesAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) {
      error_ = true;
      return;
    }

    while( gui_autoexec_.get() ) {

      // Reset incase the user quits after this iteration.
      gui_autoexec_.reset();

      // Get the next view.
      geometry = ogeom_port->getData( 0, 0, GEOM_VIEW );

      if( geometry == 0 ) {
	error("View port is no longer attached to the viewer.");
	return;    
      }

      if(  view_ != *(geometry->view)) {
	cerr << "+";

	view_ = *(geometry->view);

	if( build_field || build_geom )
	  algo->execute(fHandle, view_, build_field_, build_geom_ );

	if( build_field )
	  fHandle_ = algo->get_field();

	cerr << "-";

	if( build_geom ) {
	  // Stop showing the previous geometry.
	  if ( geomID_ ) {
	    ogeom_port->delObj( geomID_ );
	    geomID_ = 0;
	    geomflush = true;
	  }

	  GeomHandle gHandle = algo->get_geom( isoval_ );
	
	  if (gHandle.get_rep()) {
	    GeomGroup *geom = scinew GeomGroup;

	    MaterialHandle matl;
	      
	    if (have_ColorMap)
	      matl = cmHandle->lookup(isoval_);
	    else
	      matl = scinew Material(Color(gui_color_r_.get(),
					   gui_color_g_.get(),
					   gui_color_b_.get()));
	      
	    geom->add(scinew GeomMaterial( gHandle, matl ));

	    string fldname;
	    if (fHandle->get_property("name", fldname) && fldname.length() )
	      geomID_ = ogeom_port->addObj( geom, fldname );
	    else
	      geomID_ = ogeom_port->addObj( geom, string("Silhouettes") );

	    geomflush = true;
	  }
	  
	  if (geomflush) {
	    ogeom_port->flushViews();
	    geomflush = false;
	  }
	}
      }
    }

    // Just in case the geom was deleted but never recreated.
    if (geomflush) {
      ogeom_port->flushViews();
      geomflush = false;
    }
  }

  // Get a handle to the output field port.
  if ( build_field && fHandle_.get_rep() ) {
    // Get a handle to the output field port.
    FieldOPort* ofield_port = (FieldOPort *) get_oport("Silhouette Field");
  
    if (!ofield_port) {
      error("Unable to initialize oport 'Silhouette Field'.");
      return;
    }

    // Send the data downstream
    ofield_port->send(fHandle_);
  }
}

void
Silhouettes::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

CompileInfoHandle
SilhouettesAlgo::get_compile_info(const TypeDescription *ftd,
				 const TypeDescription *ttd )
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("SilhouettesAlgoT");
  static const string base_class_name("SilhouettesAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." +
		       ttd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() + "<" + ttd->get_name() + "> " + ", " +
                       "CurveField" + "<" + ttd->get_name() + "> " + ", " +
                       "CurveMesh" );
  
  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("SCIRun");
  rval->add_namespace("PCS");
  ftd->fill_compile_info(rval);
  return rval;
}

} // End namespace PCS


