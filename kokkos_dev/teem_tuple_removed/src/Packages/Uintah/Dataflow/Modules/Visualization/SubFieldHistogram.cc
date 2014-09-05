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
 * SubFieldHistogram.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Packages/Uintah/Dataflow/Modules/Visualization/SubFieldHistogram.h>

#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomSticky.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/HistogramTex.h>
#include <Core/Datatypes/LatVolField.h>

#include <Dataflow/Widgets/PointWidget.h>
#include <iostream>
#ifdef __sgi
#include <ios>
#endif
#include <algorithm>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
using std::cerr;
using std::hex;
using std::dec;

namespace Uintah {

using SCIRun::Field;
using SCIRun::LatVolField;
using SCIRun::HistogramTex;
using SCIRun::GeomGroup;
using SCIRun::GeomText;
using SCIRun::GeomSticky;
using SCIRun::Point;
using SCIRun::Color;
using SCIRun::GeomMaterial;
using SCIRun::Material;
using SCIRun::to_string;
using SCIRun::ColorMapHandle;
using SCIRun::Allocator;
using SCIRun::AuditAllocator;
using SCIRun::DumpAllocator;

using SCIRun::default_allocator;

static string control_name("Control Widget");
			 
  DECLARE_MAKER(SubFieldHistogram)

  SubFieldHistogram::SubFieldHistogram(GuiContext* ctx)
  : Module("SubFieldHistogram", ctx, Filter, "Visualization", "Uintah"),
    is_fixed_(ctx->subVar("is_fixed_")),
    min_(ctx->subVar("min_")), max_(ctx->subVar("max_"))
{
  white = scinew Material(Color(0,0,0), Color(0.6,0.6,0.6), Color(0.6,0.6,0.6), 20);
    
}

SubFieldHistogram::~SubFieldHistogram()
{

}

void SubFieldHistogram::widget_moved(bool last)
{
    if(last && !abort_flag)
    {
	abort_flag=1;
	want_to_execute();
    }
}

void SubFieldHistogram::execute(void)
{

  infield = (FieldIPort *)get_iport("Scalar Field");
  in_subfield = (FieldIPort *)get_iport("Scalar SubField");
  incolormap = (ColorMapIPort *)get_iport("Color Map");
  ogeom = (GeometryOPort *)get_oport("Geometry");

  if (!infield) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  
  if (!in_subfield) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  
  if (!incolormap) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!ogeom) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  if (!infield->get(field)) {
    postMessage("No incoming scalar field in  "+name+"'s iport\n");
    return;
  } else if (!field.get_rep()) {
    postMessage("No rep  in  "+name+"'s incoming scalar field\n");
    return;
  }

  if (!in_subfield->get(sub_field) ) {
    postMessage("No incoming scalar sub_field in  "+name+"'s iport\n");
    return;
  } else if (!sub_field.get_rep()) {
    postMessage("No rep  in  "+name+"'s incoming scalar sub_field\n");
    return;
  }
  
  ColorMapHandle cmap;
  if( !incolormap->get(cmap)){
    return;
  }

  if( field->get_type_name(0) == "double" &&
      sub_field->get_type_name(0) == "int" ){
    postMessage("Field type mismatch in "+name+" cannot make histogram\n");
    return;
  }

  bool histo_good = false;
  for(int i = 0; i < 255; i++){
    count_[i] = 0;
  }
  cerr<<"made it to dynamic cast\n";
  cerr<<field->get_type_name(0)<<" "<<field->get_type_name(1)<<"\n";
  cerr<<sub_field->get_type_name(0)<<" "<<sub_field->get_type_name(1)<<"\n";
  if(LatVolField<double> *scalarField1 =
     dynamic_cast<LatVolField<double>*>(field.get_rep())){
    if(LatVolField<int> *scalarField2 =
       dynamic_cast<LatVolField<int>*>(sub_field.get_rep())){
      cerr<<"made it inside  dynamic cast if \n";
      
      histo_good = fill_histogram( scalarField1, scalarField2 );
      cerr<<"histo is good\n";
      if( histo_good ){
	GeomGroup *all = new GeomGroup();
	double xsize = 15./16.0;
	double ysize = 0.6;
	HistogramTex *histo = new HistogramTex( Point( 0, -0.92, 0),
					     Point( xsize, -0.92, 0),
					     Point( xsize, -0.92 + ysize, 0 ),
					     Point( 0, -0.92 + ysize, 0 ));
	histo->set_buckets( count_, 256, min_i, max_i );
	histo->set_texture(cmap->rawRGBA_ );
	all->add(histo);
  
    
	// some bases for positioning text
	double xloc = xsize;
	//	double yloc = -1 + 1.1 * ysize;
	double yloc = -0.98;
  
	// create min and max numbers at the ends
	char value[80];
	sprintf(value, "%.2g", max_.get() );
	all->add( new GeomMaterial( new GeomText(value, Point(xloc,yloc,0) ),
				    white) );
	sprintf(value, "%.2g", min_.get() );
	all->add( new GeomMaterial( new GeomText(value,
						 Point(0,yloc,0)), white));
  
	// fill in 3 other places
	for(int i = 1; i < 4; i++ ) {
	  sprintf( value, "%.2g", min_.get() + i*(max_.get()-min_.get())/4.0 );
	  all->add( new GeomMaterial( new GeomText(value,
						   Point(xloc*i/4.0,yloc,0)),
				      white) );
	}
	GeomSticky *sticky = new GeomSticky(all);
	ogeom->delAll();
	ogeom->addObj(sticky, "HistogramTex Transparent" );
      }
    }
  }
}

} // End namespace Uintah

