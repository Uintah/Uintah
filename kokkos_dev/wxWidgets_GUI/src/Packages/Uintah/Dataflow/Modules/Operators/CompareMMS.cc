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

static double current_time = 0.0;

class CompareMMS : public Module {

public:
  CompareMMS( GuiContext * ctx );
  virtual ~CompareMMS();
  virtual void execute();

  static bool update_time_cb(void * module_pointer);

private:

};

DECLARE_MAKER(CompareMMS)

CompareMMS::CompareMMS(GuiContext* ctx) :
  Module("CompareMMS", ctx, Sink, "Operators", "Uintah")
{
}

CompareMMS::~CompareMMS()
{
  sched->remove_callback(update_time_cb, this);
}

void
CompareMMS::execute()
{
  printf("execute: %lf\n", current_time);

  // Much of this code comes from the SampleLattice.cc module.
  enum DataTypeEnum { SCALAR, VECTOR, TENSOR };

  Point minb, maxb;
  DataTypeEnum datatype;

  int size = 20;

  datatype = SCALAR;
  minb = Point(-size, -size, -1.0);
  maxb = Point(size, size, 1.0);

  double padpercent = 10.0;
  Vector diag((maxb.asVector() - minb.asVector()) * (padpercent/100.0));
  minb -= diag;
  maxb += diag;

  typedef ConstantBasis<double>                                     CBDBasis;
  typedef LatVolMesh< HexTrilinearLgn<Point> >                      LVMesh;
  typedef GenericField< LVMesh, CBDBasis, FData3d<double, LVMesh> > LVFieldCBD;

  // Create blank mesh.
  unsigned int sizex = size+1;//Max(2, size_x_.get());
  unsigned int sizey = size+1;//Max(2, size_y_.get());
  unsigned int sizez = 2;//Max(2, size_z_.get());

  LVMesh::handle_type mesh = scinew LVMesh(sizex, sizey, sizez, minb, maxb);

  int basis_order = 0; // cells?

  // Create Image Field.
  FieldHandle ofh;
  if (datatype == SCALAR)
  {
    if (basis_order == 0) {
      LVFieldCBD *lvf = scinew LVFieldCBD(mesh);

      //      LVFieldCBD::fdata_type &fdata = lvf->fdata();

      MMS * mms = new MMS1();

      for( int xx = 0; xx < size; xx++ ) {
        for( int yy = 0; yy < size; yy++ ) {
          for( int zz = 0; zz < 1; zz++ ) {
            LVMesh::Cell::index_type pos(mesh.get_rep(),xx,yy,zz);
            lvf->set_value( mms->pressure( xx, yy, current_time ), pos );
          }
        }
      }
      ofh = lvf;
    } else {
      error("Unsupported basis");
      return;
    }
  }
  else {
    error("Unsupported datatype.");
    return;
  }

  sched->add_callback(update_time_cb, this);


  FieldOPort *ofp = (FieldOPort *)get_oport("Comparison Field");
  ofp->send_and_dereference(ofh);
}

bool
CompareMMS::update_time_cb(void* module_pointer) {
  current_time += .10;

  if( current_time < 10000 ) {
    //    ((CompareMMS*)module_pointer)->sched->add_callback(update_time_cb,
    //                                                   (CompareMMS*)module_pointer);
    ((CompareMMS*)module_pointer)->want_to_execute();
  }

  return true;
}
