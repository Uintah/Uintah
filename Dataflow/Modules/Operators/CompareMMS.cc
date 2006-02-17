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

private:

};

DECLARE_MAKER(CompareMMS)

CompareMMS::CompareMMS(GuiContext* ctx) :
  Module("CompareMMS", ctx, Sink, "Operators", "Uintah"),
  gui_field_name_(ctx->subVar("field_name", false))
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
    warning("No input connected to the Scalar Field input port.");
    return;
  }

  if (!fh->query_scalar_interface(this).get_rep())
  {
    error("This module only works on scalar fields.");
    return;
  }

  string field_name = "Not Specified";
  fh->get_property("varname",field_name);

  gui_field_name_.set( field_name + " -- " + fh->mesh()->type_name() );

  const BBox bbox = fh->mesh()->get_bounding_box();

  cout << "BBox: " << bbox.min() << "  ----  " << bbox.max() << "\n";

  vector<unsigned int> dimensions;
  bool result = fh->mesh()->get_dim( dimensions );

  if( result ) {
    for( int pos = 0; pos < dimensions.size(); pos++ ) {
      printf("dim[%d] is: %d\n",pos,dimensions[pos]);
    }
  } else {
    printf("dimensions not returned???\n");
  }

  LVMesh* mesh = dynamic_cast<LVMesh*>(fh->mesh().get_rep());
  if( !mesh ) {
    printf("error here\n");
    error( "failed to cast mesh" );
    return;
  }

  LVMesh::Cell::index_type pos(mesh,1,1,1 );

  LVFieldCBD* field = dynamic_cast<LVFieldCBD*>(fh.get_rep());

  if( !field ) {
    printf("ERROR HERE\n");
    error( "failed to cast field" );
    return;
  }
  double val;
  field->value( val, pos );

  printf("val is %lf\n", val);

  return;
#if 0

  Point minb, maxb;

  int size = 20;

  datatype = SCALAR;
  minb = Point(-size, -size, -1.0);
  maxb = Point(size, size, 1.0);

  double padpercent = 10.0;
  Vector diag((maxb.asVector() - minb.asVector()) * (padpercent/100.0));
  minb -= diag;
  maxb += diag;

  // Create blank mesh.
  unsigned int sizex = size+1;//Max(2, size_x_.get());
  unsigned int sizey = size+1;//Max(2, size_y_.get());
  unsigned int sizez = 2;//Max(2, size_z_.get());

  LVMesh::handle_type mesh = scinew LVMesh(sizex, sizey, sizez, minb, maxb);

  int basis_order = 0; // cells?

  // Create Image Field.
  FieldHandle ofh;

  LVFieldCBD *lvf = scinew LVFieldCBD(mesh);

  MMS * mms = new MMS1();

  for( int xx = 0; xx < size; xx++ ) {
    for( int yy = 0; yy < size; yy++ ) {
      for( int zz = 0; zz < 1; zz++ ) {
        LVMesh::Cell::index_type pos(mesh.get_rep(),xx,yy,zz);
        lvf->set_value( mms->pressure( xx, yy, 0.0 ), pos );
      }
    }
  }
  ofh = lvf;

  FieldOPort *ofp = (FieldOPort *)get_oport("Scalar Field");
  ofp->send_and_dereference(ofh);
#endif
} // end execute()


