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

#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/FieldInterface.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/Scheduler.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Packages/Uintah/Core/Datatypes/Archive.h>

#include <Packages/Uintah/Dataflow/Modules/Operators/MMS/MMS.h>
#include <Packages/Uintah/Dataflow/Modules/Operators/MMS/MMS1.h>
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
  typedef ConstantBasis<double>                                     CBDBasis;
  typedef LatVolMesh< HexTrilinearLgn<Point> >                      LVMesh;
  typedef GenericField< LVMesh, CBDBasis, FData3d<double, LVMesh> > LVFieldCBD;



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
    gui->eval( id + " set_to_exact" );
    return;
  }

  if (!fh->query_scalar_interface(this).get_rep())
  {
    error("This module only works on scalar fields.");
    return;
  }

  string field_name;
  double field_time;
  fh->get_property( "varname", field_name );
  fh->get_property( "time",    field_time );

  enum field_type_e { PRESSURE, UVEL, INVALID };
  field_type_e field_type;

  if      ( field_type == "press_CC" )  field_type = PRESSURE;
  else if ( field_type == "uvel_FCME" ) field_type = UVEL;
  else {
    string msg = "MMS currently only knows how to compare pressure and uVelocity... you have: " + field_name;
    error( msg );
    return;
  }

  gui_field_name_.set( field_name );
  gui_field_time_.set( field_time );

  if( gui_output_choice_.get() == 0 ) { // Just pass original Field through

    FieldOPort *ofp = (FieldOPort *)get_oport("Scalar Field");
    ofp->send_and_dereference( fh );

  } else {

    // handle showing the exact solution or the diff

    vector<unsigned int> dimensions;
    bool result = fh->mesh()->get_dim( dimensions );

    if( !result ) {
      error("dimensions not returned???\n");
      return;
    }
    
    LVMesh* mesh = dynamic_cast<LVMesh*>(fh->mesh().get_rep());
    if( !mesh ) {
      printf("error here\n");
      error( "failed to cast mesh" );
      return;
    }
    LVFieldCBD* field = dynamic_cast<LVFieldCBD*>(fh.get_rep());
    if( !field ) {
      printf("ERROR HERE\n");
      error( "failed to cast field" );
      return;
    }
    
    Point minb, maxb;
    
    int size = 20;

    minb = Point(0,0,0);
    maxb = Point(1, 1, 1);

    // Create blank mesh.
    unsigned int sizex = dimensions[0]+1;
    unsigned int sizey = dimensions[1]+1;
    unsigned int sizez = 2;

    //printf( "size is %d, %d, %d\n", sizex, sizey, sizez );
    
    LVMesh::handle_type outputMesh = scinew LVMesh(sizex, sizey, sizez, minb, maxb);

    Transform temp;
  
    mesh->get_canonical_transform( temp );
    outputMesh->transform( temp );

    FieldHandle ofh;

    LVFieldCBD *lvf = scinew LVFieldCBD(outputMesh);

    MMS * mms = new MMS1();

    bool   showDif = (gui_output_choice_.get() == 2);
    double time = gui_field_time_.get();

    for( int xx = 0; xx < dimensions[0]-1; xx++ ) { // Not sure why I have to subtract 1 from the dimension...
      for( int yy = 0; yy < dimensions[1]-1; yy++ ) {
        for( int zz = 0; zz < 1; zz++ ) {
          LVMesh::Cell::index_type pos(outputMesh.get_rep(),xx,yy,zz);

          LVMesh::Cell::index_type inputMeshPos(mesh,xx,yy,zz);

          double calculatedValue;
          switch( field_type ) {
          case PRESSURE:
            calculatedValue = mms->pressure( xx, yy, time );
            break;
          case UVEL:
            calculatedValue = mms->uVelocity( xx, yy, time );
            break;
          }
          if( showDif ) {
            double val;
            field->value( val, inputMeshPos ); // Get the value at pos

            lvf->set_value( calculatedValue - val, pos );
          } else {
            lvf->set_value( calculatedValue, pos );
          }
        }
      }

    } 
    ofh = lvf;
    
    FieldOPort *ofp = (FieldOPort *)get_oport("Scalar Field");
    ofp->send_and_dereference(ofh);
  } // end if gui_output_choice_ == 0;

} // end execute()

