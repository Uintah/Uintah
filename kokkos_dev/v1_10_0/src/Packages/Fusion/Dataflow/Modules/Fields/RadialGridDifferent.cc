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

#include <Core/Datatypes/HexVolField.h>

#include <Core/Geometry/Vector.h>

#include <Packages/Fusion/share/share.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE RadialGridDifferent : public Module {
public:
  RadialGridDifferent(GuiContext *context);

  virtual ~RadialGridDifferent();

  virtual void execute();

protected:
  FieldHandle fHandle_;

  int fGeneration_;

  int idim_;
  int jdim_;
  int kdim_;
};


DECLARE_MAKER(RadialGridDifferent)


RadialGridDifferent::RadialGridDifferent(GuiContext *context)
  : Module("RadialGridDifferent", context, Source, "Fields", "Fusion"),

    fGeneration_( -1 ),

    idim_(0),
    jdim_(0),
    kdim_(0)
{
}

RadialGridDifferent::~RadialGridDifferent(){
}

void RadialGridDifferent::execute(){

  // Get a handle to the input field port.
  FieldHandle  fHandle;

  FieldIPort* ifield_port = (FieldIPort *) get_iport("Input Field");

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

  // Get the current HexVolMesh.
  HexVolMesh *hvm = NULL;

  if( (hvm = dynamic_cast<HexVolMesh*> (fHandle->mesh().get_rep()) ) ) {

    // Get the dimensions of the mesh.
    if( !hvm->get_property( "I Dim", idim_ ) ||
	!hvm->get_property( "J Dim", jdim_ ) ||
	!hvm->get_property( "K Dim", kdim_ ) ) {
      error( "No Mesh Dimensions." );
      return;
    }

    HexVolMesh::Node::size_type npts;

    hvm->size( npts );

    // Make sure they match the number of points in the mesh.
    if( npts != idim_ * jdim_ * kdim_ ) {
      error( "Mesh dimensions do not match mesh size." );
      return;
    }

    // If no data or a changed recreate the mesh.
    if( !fHandle_.get_rep() ||
	fGeneration_ != fHandle->generation ) {
      fGeneration_ = fHandle->generation;

      HexVolField<double> *hvf = 
	scinew HexVolField<double>(HexVolMeshHandle(hvm), Field::NODE);

      fHandle_ = FieldHandle( hvf );

      // Add the data to the field.
      HexVolField<double>::fdata_type::iterator out = hvf->fdata().begin();

      Vector vec;

      // Get the length between nodes in the radial direction
      for( int i=0; i<npts; i++ )
      {
	// Skip the first point.
	if( i % idim_ == 0 ) {
	  *out = 0;
	  out++;
	}
	else {
	  vec = Vector( hvm->point( i ) - hvm->point( i-1 ) );

	  *out = vec.length();
	  out++;
	}
      }
    }
  }
  else {
    error( "Only availible for HexVol Mesh data" );
    return;
  }

  // Get a handle to the output Field port.
  if( fHandle_.get_rep() )
  {
    FieldOPort *ofield_port =
      (FieldOPort *) get_oport("Output Radial Diff. Field");

    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( fHandle_ );
  }
}


} // End namespace Fusion
