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
 *  RadialGridDifferent.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   Febuary 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>

#include <Core/Datatypes/ColumnMatrix.h>

#include <Core/Datatypes/HexVolField.h>

#include <Core/Geometry/Vector.h>

#include <Packages/Fusion/share/share.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE RadialGridDifferent : public Module {
public:
  RadialGridDifferent(const string& id);

  virtual ~RadialGridDifferent();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);

protected:
  MatrixHandle mHandle_;

  int generation_;

  int idim_;
  int jdim_;
  int kdim_;
};

extern "C" FusionSHARE Module* make_RadialGridDifferent(const string& id) {
  return scinew RadialGridDifferent(id);
}

RadialGridDifferent::RadialGridDifferent(const string& id)
  : Module("RadialGridDifferent", id, Source, "Fields", "Fusion"),

    generation_( -1 ),

    idim_(0),
    jdim_(0),
    kdim_(0)
{
}

RadialGridDifferent::~RadialGridDifferent(){
}

void RadialGridDifferent::execute(){

  bool update = false;

  // Get a handle to the input field port.
  FieldHandle  fHandle;

  FieldIPort* ifield_port =
    (FieldIPort *)	get_iport("Input Grid Field");

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


  // Check to see if the input data has changed.
  if( generation_ != fHandle->generation ) {
    generation_  = fHandle->generation;

    cout << "MeshBuilder - New Data." << endl;

    update = true;
  }

  // Get the current HexVolMesh.
  HexVolMesh *hvmInput = (HexVolMesh*) (fHandle->mesh().get_rep());


  // Get the dimensions of the mesh.
  if( !hvmInput->get_property( "I Dim", idim_ ) ||
      !hvmInput->get_property( "J Dim", jdim_ ) ||
      !hvmInput->get_property( "K Dim", kdim_ ) ) {
    error( "No Mesh Dimensions." );
    return;
  }


  HexVolMesh::Node::size_type npts;

  hvmInput->size( npts );

  // Make sure they match the number of points in the mesh.
  if( npts != idim_ * jdim_ * kdim_ ) {
    error( "Mesh dimensions do not match mesh size." );
    return;
  }

  // Create a new HexVolMesh based on the old points;
  if( !mHandle_.get_rep() || update ) {

    ColumnMatrix *dMatrix = scinew ColumnMatrix(npts);

    Vector vec;

    double dRad;

    // Get the length between nodes in the radial direction
    for( int i=0; i<npts; i++ )
    {
      // Skip the first point.
      if( i % idim_ == 0 ) {
	dMatrix->put(i, 0, 0.0);
      }
      else {

	vec = Vector( hvmInput->point( i ) - hvmInput->point( i-1 ) );
	dRad = vec.length() * 100.0;

	dMatrix->put(i, 0, dRad );
      }

    }

    mHandle_ = MatrixHandle( dMatrix );
  }

  // Get a handle to the output V Field matrix port.
  {
    MatrixOPort *omatrix_port =
      (MatrixOPort *) get_oport("Output Radial Diff.");

    if (!omatrix_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    omatrix_port->send( mHandle_ );
  }
}

void RadialGridDifferent::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Fusion
