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
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomSticky.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/HistogramTex.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Widgets/PointWidget.h>
#include <iostream>
#ifdef __sgi
#  include <ios>
#endif
#include <algorithm>

using std::cerr;
using std::hex;
using std::dec;

namespace Uintah {

using SCIRun::Field;
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
    is_fixed_(get_ctx()->subVar("is_fixed_")),
    min_(get_ctx()->subVar("min_")), max_(get_ctx()->subVar("max_"))
{
  white = scinew Material(Color(0,0,0), Color(0.6,0.6,0.6), Color(0.6,0.6,0.6), 20);
    
}

SubFieldHistogram::~SubFieldHistogram()
{

}

void
SubFieldHistogram::widget_moved(bool last,BaseWidget*)
{
  if(last && !abort_flag_)
    {
      abort_flag_ = true;
      want_to_execute();
    }
}

void
SubFieldHistogram::execute(void)
{

  infield = (FieldIPort *)get_iport("Scalar Field");
  in_subfield = (FieldIPort *)get_iport("Scalar SubField");
  incolormap = (ColorMapIPort *)get_iport("Color Map");
  ogeom = (GeometryOPort *)get_oport("Geometry");

  if (!infield) {
    error("Unable to initialize " + module_name_ + "'s iport.");
    return;
  }
  
  if (!in_subfield) {
    error("Unable to initialize " + module_name_ + "'s iport.");
    return;
  }
  
  if (!incolormap) {
    error("Unable to initialize " + module_name_ + "'s iport.");
    return;
  }
  if (!ogeom) {
    error("Unable to initialize " + module_name_ + "'s oport.");
    return;
  }

  if (!infield->get(field)) {
    error("No incoming scalar field in " + module_name_ + "'s iport.");
    return;
  } else if (!field.get_rep()) {
    error("No rep in " + module_name_ + "'s incoming scalar field.");
    return;
  }

  if (!in_subfield->get(sub_field) ) {
    error("No incoming scalar sub_field in " + module_name_ + "'s iport.");
    return;
  } else if (!sub_field.get_rep()) {
    error("No rep in " + module_name_ + "'s incoming scalar sub_field.");
    return;
  }
  
  ColorMapHandle cmap;
  if( !incolormap->get(cmap)){
    return;
  }

  const TypeDescription *td = field->get_type_description();
  const TypeDescription *std = sub_field->get_type_description();
  if( td->get_name().find("double") != string::npos &&
      std->get_name().find("int") != string::npos)
  {
    error("Field type mismatch in "+ module_name_ +", cannot make histogram.");
    return;
  }

  bool histo_good = false;
  for(int i = 0; i < 255; i++){
    count_[i] = 0;
  }

  CDField *cdfld = dynamic_cast<CDField*>(field.get_rep());
  CIField *cifld = dynamic_cast<CIField*>(sub_field.get_rep());
  LDField *ldfld = 0;
  LIField *lifld = 0;
  
  if( !cdfld || !cifld ){
    ldfld = dynamic_cast<LDField*>(field.get_rep());
    lifld = dynamic_cast<LIField*>(sub_field.get_rep());
    if( !ldfld || !lifld ){
      error("dynamic cast failed");
      return;
    } else {
      histo_good = fill_histogram( ldfld, lifld );
    }
  } else {
    histo_good = fill_histogram( cdfld, cifld );
  }
  if( histo_good ){
    GeomGroup *all = new GeomGroup();
    double xsize = 15./16.0;
    double ysize = 0.6;
    HistogramTex *histo = new HistogramTex( Point( 0, -0.92, 0),
                                            Point( xsize, -0.92, 0),
                                            Point( xsize, -0.92 + ysize, 0 ),
                                            Point( 0, -0.92 + ysize, 0 ));
    histo->set_buckets( count_, 256, min_i, max_i );
    histo->set_texture(cmap->get_rgba());
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

} // End namespace Uintah

