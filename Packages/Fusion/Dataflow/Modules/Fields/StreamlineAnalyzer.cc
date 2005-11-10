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
 *  StreamlineAnalyzer.cc:
 *
 *  Written by:
 *   Allen R. Sanderson
 *   SCI Institute
 *   University of Utah
 *   September 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/Handle.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/StructQuadSurfField.h>

#include <Packages/Fusion/Dataflow/Modules/Fields/StreamlineAnalyzer.h>

namespace Fusion {

using namespace SCIRun;

class StreamlineAnalyzer : public Module
{
public:
  StreamlineAnalyzer(GuiContext* ctx);
  virtual ~StreamlineAnalyzer();

  virtual void execute();

protected:
  GuiString gPlanesStr_;
  GuiInt gPlanesInt_;
  GuiInt gColor_;
  GuiInt gMaxWindings_;
  GuiInt gOverride_;
  GuiInt gCurveMesh_;
  GuiInt gScalarField_;
  GuiInt gShowIslands_;
  GuiInt gOverlaps_;

  vector< double > planes_;

  unsigned int color_;
  unsigned int maxWindings_;
  unsigned int override_;
  unsigned int curveMesh_;
  unsigned int scalarField_;
  unsigned int showIslands_;
  unsigned int overlaps_;

  FieldHandle slfieldout_;
  FieldHandle pccfieldout_;
  FieldHandle pcsfieldout_;

  int slfGeneration_;
  int pccfGeneration_;
  int pcsfGeneration_;
};


DECLARE_MAKER(StreamlineAnalyzer)

StreamlineAnalyzer::StreamlineAnalyzer(GuiContext* context)
  : Module("StreamlineAnalyzer", context, Source, "Fields", "Fusion"),
    gPlanesStr_(context->subVar("planes-list")),
    gPlanesInt_(context->subVar("planes-quantity")),
    gColor_(context->subVar("color")),
    gMaxWindings_(context->subVar("maxWindings")),
    gOverride_(context->subVar("override")),
    gCurveMesh_(context->subVar("curve-mesh")),
    gScalarField_(context->subVar("scalar-field")),
    gShowIslands_(context->subVar("show-islands")),
    gOverlaps_(context->subVar("overlaps")),
    color_(1),
    maxWindings_(30),
    override_(0),
    curveMesh_(1),
    scalarField_(1),
    showIslands_(0),
    overlaps_(0),
    slfGeneration_(-1),
    pccfGeneration_(-1),
    pcsfGeneration_(-1)
{
}

StreamlineAnalyzer::~StreamlineAnalyzer()
{
}

void
StreamlineAnalyzer::execute()
{
  bool update = false;

  cerr << "StreamlineAnalyzer getting ports " << endl;

  FieldIPort* ifp = (FieldIPort *)get_iport("Input Streamlines");
  FieldHandle slfieldin;

  cerr << "StreamlineAnalyzer getting field " << endl;

  if (!(ifp->get(slfieldin))) {
    error( "No handle or representation." );
    return;
  }

  cerr << "StreamlineAnalyzer getting field rep" << endl;

  if (!(slfieldin.get_rep())) {
    error( "No handle or representation." );
    return;
  }

  cerr << "StreamlineAnalyzer getting type " << endl;

  if (slfieldin->get_type_description(0)->get_name() != "CurveField" &&
      slfieldin->get_type_description(0)->get_name() != "StructCurveField") {
    error("Only available for (Struct)CurveFields.");
    return;
  }

  cerr << "StreamlineAnalyzer getting interface " << endl;

  if (!slfieldin->query_scalar_interface(this).get_rep()) {
    error("Only available for Scalar data.");
    return;
  }

  // Check to see if the input field has changed.
  if( slfGeneration_ != slfieldin->generation ) {
    slfGeneration_ = slfieldin->generation;
    update = true;
  }


  // Get a handle to the input centroid field port.
  ifp = (FieldIPort *) get_iport("Input Centroids");
  FieldHandle pccfieldin;

  // The field input is optional.
  if (ifp->get(pccfieldin) && pccfieldin.get_rep()) {
    
    if (pccfieldin->get_type_description(0)->get_name() != "PointCloudField" &&
	pccfieldin->get_type_description(1)->get_name() != "double") {
      error("Only available for Point Cloud Fields of type double.");
      return;
    }

    if (!pccfieldin->query_scalar_interface(this).get_rep()) {
      error("Only available for Scalar data.");
      return;
    }

    // Check to see if the input field has changed.
    if( pccfGeneration_ != pccfieldin->generation ) {
      pccfGeneration_ = pccfieldin->generation;
      update = true;
    }

  } else {
    pccfGeneration_ = -1;
  }

  // Get a handle to the input separatrices field port.
  ifp = (FieldIPort *) get_iport("Input Separatrices");
  FieldHandle pcsfieldin;

  // The field input is optional.
  if (ifp->get(pcsfieldin) && pcsfieldin.get_rep()) {
    
    if (pcsfieldin->get_type_description(0)->get_name() != "PointCloudField" &&
	pcsfieldin->get_type_description(1)->get_name() != "double") {
      error("Only available for Point Cloud Fields of type double.");
      return;
    }

    if (!pcsfieldin->query_scalar_interface(this).get_rep()) {
      error("Only available for Scalar data.");
      return;
    }

    // Check to see if the input field has changed.
    if( pcsfGeneration_ != pcsfieldin->generation ) {
      pcsfGeneration_ = pcsfieldin->generation;
      update = true;
    }

  } else {
    pcsfGeneration_ = -1;
  }


  cerr << "StreamlineAnalyzer getting gui " << endl;

  vector< double > planes(0);

  if( gPlanesInt_.get() ) {

    unsigned int nplanes = gPlanesInt_.get();

    for( unsigned int i=0; i<nplanes; i++ )
      planes.push_back(2.0 * M_PI * (double) i / (double) nplanes );

  } else {

    istringstream plist(gPlanesStr_.get());
    double plane;
    while(!plist.eof()) {
      plist >> plane;
      if (plist.fail()) {
	if (!plist.eof()) {
	  plist.clear();
	  warning("List of Planes was bad at character " +
		  to_string((int)(plist.tellg())) +
		  "('" + ((char)(plist.peek())) + "').");
	}
	break;

      } else if (!plist.eof() && plist.peek() == '%') {
	plist.get();
	plane = 0 + (2.0*M_PI - 0) * plane / 100.0;
      }

      if( 0 <= plane && plane <= 2.0*M_PI )
	planes.push_back(plane);
      else {
	error("Plane is not in the range of 0 to 2 PI.");
	return;
      }
    }
  }


  if( planes_.size() != planes.size() ){
    update = true;

    planes_.resize(planes.size());

    for( unsigned int i=0; i<planes.size(); i++ )
      planes_[i] = planes[i];

  } else {
    for( unsigned int i=0; i<planes.size(); i++ ) {
      if( fabs( planes_[i] - planes[i] ) > 1.0e-4 ) {
	planes_[i] = planes[i];
	update = true;
      }
    }
  }

  if( color_ != (unsigned int) gColor_.get() ) {
    update = true;

    color_ = gColor_.get();
  }

  if( maxWindings_ != (unsigned int) gMaxWindings_.get() ) {
    update = true;

    maxWindings_ = gMaxWindings_.get();
  }

  if( override_ != (unsigned int) gOverride_.get() ) {
    update = true;

    override_ = gOverride_.get();
  }

  if( curveMesh_ != (unsigned int) gCurveMesh_.get() ) {
    update = true;

    curveMesh_ = gCurveMesh_.get();
  }

  if( scalarField_ != (unsigned int) gScalarField_.get() ) {
    update = true;

    scalarField_ = gScalarField_.get();
  }

  if( showIslands_ != (unsigned int) gShowIslands_.get() ) {
    update = true;

    showIslands_ = gShowIslands_.get();
  }

  if( overlaps_ != (unsigned int) gOverlaps_.get() ) {
    update = true;

    overlaps_ = gOverlaps_.get();
  }

  cerr << "StreamlineAnalyzer executing " << endl;

  // If no data or a changed recalcute.
  if( update ||
      !slfieldout_.get_rep()) {

    Vector* tmpvec;

    const TypeDescription *ftd = slfieldin->get_type_description();
    const TypeDescription *ttd = scalarField_ ?
      slfieldin->get_type_description(1) :
      get_type_description(tmpvec);

    const string otd = curveMesh_ ?
      CurveField         <double>::type_name(0) : 
      StructQuadSurfField<double>::type_name(0);

    CompileInfoHandle ci =
      StreamlineAnalyzerAlgo::get_compile_info(ftd, ttd, otd);

    Handle<StreamlineAnalyzerAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    vector< pair< unsigned int, unsigned int > > topology;

    algo->execute(slfieldin, slfieldout_,
		  pccfieldin, pccfieldout_,
		  pcsfieldin, pcsfieldout_,
		  planes_,
		  color_, showIslands_, overlaps_,
		  maxWindings_, override_, topology);
  }

  cerr << "StreamlineAnalyzer sending data " << endl;

  // Get a handle to the output field port.
  if ( slfieldout_.get_rep() ) {
    FieldOPort* ofp = (FieldOPort *) get_oport("Output Poincare");

    // Send the data downstream
    ofp->send(slfieldout_);
  }

  if ( pccfieldout_.get_rep() ) {
    FieldOPort* ofp = (FieldOPort *) get_oport("Output Centroids");

    // Send the data downstream
    ofp->send(pccfieldout_);
  }

  if ( pcsfieldout_.get_rep() ) {
    FieldOPort* ofp = (FieldOPort *) get_oport("Output Separatrices");

    // Send the data downstream
    ofp->send(pcsfieldout_);
  }

  cerr << "StreamlineAnalyzer done " << endl;
}


CompileInfoHandle
StreamlineAnalyzerAlgo::get_compile_info(const TypeDescription *ftd,
					 const TypeDescription *ttd,
					 const string otd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));

  const string template_class_name = (ttd->get_name() == string("Vector") ?
				      string("StreamlineAnalyzerAlgoVector") :
				      string("StreamlineAnalyzerAlgoScalar"));

  static const string base_class_name("StreamlineAnalyzerAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." +
		       otd + ttd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() + ", " +
                       otd + "<" + ttd->get_name() + "> , " +
                       "PointCloudField< double > , " +
		       ttd->get_name() );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  rval->add_namespace("Fusion");
  return rval;
}

} // End namespace Fusion
