/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  MDSPlusDataReader.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   August 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Containers/Handle.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Ports/FieldPort.h>

#include <Packages/Fusion/Dataflow/Modules/Render/Plot2DViewer.h>

#include <sci_defs.h>

#ifdef HAVE_TCLMATRIX
#include <plplot/tclMatrix.h>
#endif

namespace Fusion {

using namespace SCIRun;


extern "C" FusionSHARE Module* make_Plot2DViewer(GuiContext* ctx) {
  return scinew Plot2DViewer(ctx);
}

Plot2DViewer::Plot2DViewer(GuiContext *context)
  : Module("Plot2DViewer", context, Source, "Render", "Fusion"),

    ctx(context),

    havePLplot_(context->subVar("havePLplot")),
 
    updateType_(context->subVar("updateType")),

    nPlots_(context->subVar("nPlots")),
    nData_(context->subVar("nData")),

    xMin_(context->subVar("xmin")),
    xMax_(context->subVar("xmax")),
    yMin_(context->subVar("ymin")),
    yMax_(context->subVar("ymax")),
    zMin_(context->subVar("zmin")),
    zMax_(context->subVar("zmax")),

    updateGraph_( false ),
    ndata_(0)
{
}

Plot2DViewer::~Plot2DViewer(){
}

void Plot2DViewer::execute(){

  bool updateField  = false;
  bool updateAll    = false;

  FieldHandle fHandle;

  unsigned int ndata, idim, jdim, kdim;

  port_range_type range = get_iports("Input Field");

  if (range.first == range.second)
    return;

  port_map_type::iterator pi = range.first;
  port_map_type::iterator pe = range.second;

  pe--; // Last port should be enpty so ignore it.

  for (ndata=0; pi!=pe; ndata++, pi++) {

    // Get a handle to the input field port.
    FieldIPort *ifield_port = (FieldIPort *) get_iport(pi->second);

    if (!ifield_port) {
      error( "Unable to initialize "+name+"'s iport" );
      return;
    }

    // The field input is required.
    if (!ifield_port->get(fHandle) || !(fHandle.get_rep()) ||
	!(fHandle->mesh().get_rep())) {
      error( "No handle or representation" );
      return;
    }

    if (!fHandle->query_scalar_interface().get_rep()) {
      error( "Only availible for Scalar data." );
      return;
    }

    if( fHandle->get_type_description(0)->get_name() == "LatVolField" ||
	fHandle->get_type_description(0)->get_name() == "StructHexVolField" ) {
      LatVolMesh *lvmInput = (LatVolMesh*) fHandle->mesh().get_rep();

      idim = lvmInput->get_ni();
      jdim = lvmInput->get_nj();
      kdim = lvmInput->get_nk();
    } else if( fHandle->get_type_description(0)->get_name() == "ImageField" ||
               fHandle->get_type_description(0)->get_name() == "StructQuadSurfField" ) {
      ImageMesh *imInput = (ImageMesh*) fHandle->mesh().get_rep();

      idim = imInput->get_ni();
      jdim = imInput->get_nj();
      kdim = 1;
    } else if( fHandle->get_type_description(0)->get_name() == "ScanlineField" ||
               fHandle->get_type_description(0)->get_name() == "StructCurveField" ) {
      ScanlineMesh *slmInput = (ScanlineMesh*) fHandle->mesh().get_rep();

      idim = slmInput->get_ni();
      jdim = 1;
      kdim = 1;
    } else {
      error( fHandle->get_type_description(0)->get_name() );
      error( "Only availible for regular topological data." );
      return;
    }

    if( fGeneration_.size() < ndata+1 ) {

      fGeneration_.push_back( - 1 );
      fHandle_.push_back( fHandle );

      idim_.push_back( idim );
      jdim_.push_back( jdim );
      kdim_.push_back( kdim );
//      needsUpdated_.push_back( true );

      // Add the data in the GUI.
      {
	ostringstream str;
	str << id << " add_data " << ndata << " ";
	str << idim << " ";
	str << jdim << " ";
	str << kdim;

	gui->execute(str.str().c_str());

	remark( str.str() );
      }

      updateField  = true;
      updateAll  = true;
    }

    // Check to see if the input field has changed.
    if( fGeneration_[ndata] != fHandle->generation ) {
      fHandle_[ndata] = fHandle;
      fGeneration_[ndata] = fHandle->generation;
      updateField  = true;
      //      needsUpdated_[nData] = true;
    }

    // Check to see if the dimensions have changed.
    if( idim_[ndata] != idim ||
	jdim_[ndata] != jdim ||
	kdim_[ndata] != kdim ) {

      // Get the dimensions of the mesh.
      idim_[ndata] = idim;
      jdim_[ndata] = jdim;
      kdim_[ndata] = kdim;

      // Update the dims in the GUI.
      ostringstream str;
      str << id << " set_size " << ndata << " ";
      str << idim_[ndata] << " " << jdim_[ndata] << " " << kdim_[ndata];

      gui->execute(str.str().c_str());

      updateAll  = true;
//      needsUpdated_[nData] = true;
    }
  }
  
  if( ndata_ > ndata ) {
      while( ndata_ > ndata ) {
	fGeneration_.pop_back();
	fHandle_.pop_back();
	idim_.pop_back();
	jdim_.pop_back();
	kdim_.pop_back();

	ndata_--;
      }

      // Update the data listing
      ostringstream str;
      str << id << " data_size " << ndata_;
      gui->execute(str.str().c_str());
  }

  else if( ndata_ < ndata ) {
    ndata_ = ndata;
    
    // Update the data listing
    ostringstream str;
    str << id << " data_size " << ndata_;
    gui->execute(str.str().c_str());
  }

  if( updateField || updateAll || updateGraph_ ) {
    // Update the data listing
    ostringstream str;
    str << id << " graph_data ";
    gui->execute(str.str().c_str());

    updateGraph_ = false;
  }
}


void Plot2DViewer::trueExecute( unsigned int port, unsigned int slice )
{
  FieldHandle fHandle = fHandle_[port];

  // The field input is required.
  if ( !(fHandle.get_rep()) ||
       !(fHandle->mesh().get_rep())) {
    error( "No handle or representation" );
    return;
  }

  if (!fHandle->query_scalar_interface().get_rep()) {
    error( "Only availible for Scalar data." );
    return;
  }

  dMat_x_ = NULL;
  dMat_y_ = NULL;
  dMat_v_ = NULL;

  // If no data or a changed recreate the mesh.
  if( 1 ) {
    const TypeDescription *ftd = fHandle->get_type_description(0);
    const TypeDescription *ttd = fHandle->get_type_description(1);

    CompileInfoHandle ci = Plot2DViewerAlgo::get_compile_info(ftd,ttd);
    Handle<Plot2DViewerAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    algo->execute(fHandle, slice, this);
  }
}



void Plot2DViewer::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("Plot2DViewer needs a minor command");
    return;
  }

  if (args[1] == "have_PLplot") {
#ifdef HAVE_PLPLOT
    havePLplot_.set( 1 );
#else
    havePLplot_.set( 0 );
#endif
  }
  else if (args[1] == "update_graph") {
    updateGraph_ = true;
  }
  else if (args[1] == "vertex_coords") {
#ifdef HAVE_TCLMATRIX
    unsigned int port = atoi( args[2].c_str() );

    if( fGeneration_[port] != -1 &&
	idim_[port] > 0 &&
	jdim_[port] > 0 &&
	kdim_[port] > 0 )
    {
      double x, y, z;

      double xmin=+1.0e16, ymin=+1.0e16, zmin=+1.0e16;
      double xmax=-1.0e16, ymax=-1.0e16, zmax=-1.0e16;

      unsigned int slice = atoi( args[3].c_str() );

      trueExecute( port, slice );

      if( dMat_x_ && dMat_y_ && dMat_v_ ) {

	ostringstream str;
	str << id << " have_data 1";
	gui->execute(str.str().c_str());
 

	tclMatrix *matxg = Tcl_GetMatrixPtr( NULL, (char*) args[4].c_str() );
	tclMatrix *matyg = Tcl_GetMatrixPtr( NULL, (char*) args[5].c_str() );
	tclMatrix *matzg = Tcl_GetMatrixPtr( NULL, (char*) args[6].c_str() );

	TclMatFloat txg( matxg ), tyg( matyg ) , tzg( matzg );

	for( unsigned int i=0; i<idim_[port]; i++ ) {
	  for( unsigned int j=0; j<jdim_[port]; j++ ) {

	    x = dMat_x_->get(i,j);
	    y = dMat_y_->get(i,j);
	    z = dMat_v_->get(i,j);

	    txg(i,j) = x;
	    tyg(i,j) = y;
	    tzg(i,j) = z;

	    if( xmin > x ) xmin = x;	  if( xmax < x ) xmax = x;
	    if( ymin > y ) ymin = y;	  if( ymax < y ) ymax = y;
	    if( zmin > z ) zmin = z;	  if( zmax < z ) zmax = z;
	  }
	}

	xMin_.set( xmin );
	xMax_.set( xmax );
	yMin_.set( ymin );
	yMax_.set( ymax );
	zMin_.set( zmin );
	zMax_.set( zmax );
      }
    }
#endif

  } else if (args[1] == "add_GUIVar_Int") {
    
    scinew GuiInt(ctx->subVar(args[2]), atoi(args[3].c_str()) );

  } else if (args[1] == "add_GUIVar_String") {
    
    scinew GuiString(ctx->subVar(args[2]), args[3].c_str() );

  } else if (args[1] == "remove_GUIVar") {
    
    ctx->erase( args[2] );

  } else {
    Module::tcl_command(args, userdata);
  }
}


CompileInfoHandle
Plot2DViewerAlgo::get_compile_info(const TypeDescription *ftd,
				   const TypeDescription *ttd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("Plot2DViewerAlgoT");
  static const string base_class_name("Plot2DViewerAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." +
		       ttd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() + ", " +
		       ttd->get_name() );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("Fusion");
  ftd->fill_compile_info(rval);
  return rval;
}

} // End namespace Fusion


