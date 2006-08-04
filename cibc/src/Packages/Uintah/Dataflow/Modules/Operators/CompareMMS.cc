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

//    File   : CompareMMS.cc
//    Author : J. Davison de St. Germain
//    Date   : Jan 2006

#include <Packages/Uintah/Dataflow/Modules/Operators/CompareMMS.h>

#include <Core/Datatypes/FieldInterface.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/Scheduler.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <Packages/Uintah/Core/Datatypes/Archive.h>

#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>

#include <sgi_stl_warnings_off.h>
#include <map>
#include <iostream>
#include <sstream>
#include <sgi_stl_warnings_on.h>

using namespace std;
using namespace SCIRun;
using namespace Uintah;

class CompareMMS : public Module {

public:
  CompareMMS( GuiContext * ctx );
  virtual ~CompareMMS();
  virtual void execute();

private:
  GuiString gui_field_name_;
  GuiDouble gui_field_time_;
  GuiInt    gui_output_choice_;  // 0 == original, 1 == exact solution, 2 == difference

private:

};

DECLARE_MAKER(CompareMMS)

CompareMMS::CompareMMS(GuiContext* ctx) :
  Module("CompareMMS", ctx, Sink, "Operators", "Uintah"),
  gui_field_name_(ctx->subVar("field_name", false)),
  gui_field_time_(ctx->subVar("field_time", false)),
  gui_output_choice_(ctx->subVar("output_choice", false))
{
}

CompareMMS::~CompareMMS()
{
}

void
CompareMMS::execute()
{
//   typedef ConstantBasis<double>                                     CBDBasis;
//   typedef LatVolMesh< HexTrilinearLgn<Point> >                      LVMesh;
//   typedef GenericField< LVMesh, CBDBasis, FData3d<double, LVMesh> > LVFieldCBD;



  FieldIPort *iport = (FieldIPort*)get_iport("Scalar Field");
  if (!iport)
  {
    error("Error: unable to find (in xml file, I think) module input port named 'Scalar Field'");
    return;
  }

  // The input port (with data) is required.
  FieldHandle fh;
  if (!iport->get(fh) || !fh.get_rep())
  {
    remark("No input connected to the Scalar Field input port.");
    remark("Displaying exact solution.");
    get_gui()->eval( get_id() + " set_to_exact" );
    return;
  }

  if (!fh->query_scalar_interface(this).get_rep())
  {
    error("This module only works on scalar fields.");
    return;
  }

  bool   found_properties;

  string field_name;
  double field_time;

  found_properties = fh->get_property( "name", field_name );
  found_properties = fh->get_property( "time",    field_time );

  Point spacial_min, spacial_max;
  found_properties = fh->get_property( "spacial_min", spacial_min );
  found_properties = fh->get_property( "spacial_max", spacial_max );

  cout << "field range is: " << spacial_min << " to " << spacial_max << "\n";

  IntVector field_offset;
  found_properties = fh->get_property( "offset", field_offset );

  cout << "offset is " << field_offset << "\n";

  if( !found_properties ) {
    cout << "This field did not include all the properties I expected...\n";
  }

  CompareMMSAlgo::compare_field_type field_type;

  if      ( field_name == "press_CC" )  field_type = CompareMMSAlgo::PRESSURE;
  else if ( field_name == "vel_CC:1" ) field_type = CompareMMSAlgo::UVEL;
  else if ( field_name == "vel_CC:2" ) field_type = CompareMMSAlgo::VVEL;
  else {
    string msg = "MMS currently only knows how to compare pressure and uVelocity... you have: " + field_name;
    field_type = CompareMMSAlgo::INVALID;
    error( msg );
    return;
  }

  gui_field_name_.set( field_name );
  gui_field_time_.set( field_time );

  if( gui_output_choice_.get() == 0 ) { // Just pass original Field through
    
    FieldOPort *ofp = (FieldOPort *)get_oport("Scalar Field");
    ofp->send_and_dereference( fh );
    
  } else {
    
    const SCIRun::TypeDescription *td = fh->get_type_description();
    CompileInfoHandle ci = CompareMMSAlgo::get_compile_info(td);
    LockingHandle<CompareMMSAlgo> algo;
    if(!module_dynamic_compile(ci, algo)) {
      error("CompareMMS cannot work on this Field: dynamic compilation failed.");
      return;
    }
    
    // handle showing the exact solution or the diff
    
    vector<unsigned int> dimensions;
    bool result = fh->mesh()->get_dim( dimensions );
    
    if( !result ) {
      error("dimensions not returned???\n");
      return;
    }
    
    
    FieldHandle ofh = algo->compare(fh, dimensions,
                                    field_type,
                                    field_name, field_time,
                                    gui_output_choice_.get(),
                                    gui_field_time_.get());
      
      
    IntVector offset(0,0,0);        
    string property_name = "offset";
    fh->get_property( property_name, offset);
    ofh->set_property(property_name.c_str(), IntVector(offset) , true);
    string prefix = "Exact_";
    if ( gui_output_choice_.get() == 2) prefix = "Diff_";
    ofh->set_property("varname", string(prefix+field_name.c_str()), true);
    
    FieldOPort *ofp = (FieldOPort *)get_oport("Scalar Field");
    ofp->send_and_dereference(ofh);
  } // end if gui_output_choice_ == 0;
}
  

CompileInfoHandle
CompareMMSAlgo::get_compile_info(const SCIRun::TypeDescription *td)
{
  string subname;
  string subinc;
  string sname = td->get_name("", "");
  
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("CompareMMSAlgoT");
  static const string base_class_name("CompareMMSAlgo");
  
  if(sname.find("LatVol") != string::npos ){
    subname.append(td->get_name());
    subinc.append(include_path);
  } else {
    cerr<<"Unsupported Geometry, needs to be of Lattice type.\n";
    subname.append("Cannot compile this unupported type");
  }
  CompileInfo *rval =
    scinew CompileInfo(template_class_name + "." +
		       td->get_filename() + ".",
                       base_class_name,
                       template_class_name,
                       td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_include(subinc);
  rval->add_namespace("Uintah");
  td->fill_compile_info(rval);
  return rval;


}


