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
 *  VectorMagnitude.cc:
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

#include <Dataflow/Ports/MatrixPort.h>

#include <Core/Datatypes/ColumnMatrix.h>

#include <Packages/Fusion/share/share.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE VectorMagnitude : public Module {
public:
  VectorMagnitude(const string& id);

  virtual ~VectorMagnitude();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);

private:
  MatrixHandle mHandle_;

  int generation_;
};

extern "C" FusionSHARE Module* make_VectorMagnitude(const string& id) {
  return scinew VectorMagnitude(id);
}

VectorMagnitude::VectorMagnitude(const string& id)
  : Module("VectorMagnitude", id, Source, "Fields", "Fusion"),

    generation_( -1 )

{
}

VectorMagnitude::~VectorMagnitude(){
}

void VectorMagnitude::execute(){

  bool update = false;

  MatrixHandle mHandle;

  Matrix* vMatrix;

  // Get a handle to the input matrix port.
  MatrixIPort* imatrix_port =
    (MatrixIPort *)	get_iport("Input Vector Matrix");

  if (!imatrix_port) {
    error( "Unable to initialize "+name+"'s iport" );
    return;
  }

  // The matrix input is required.
  if (!imatrix_port->get(mHandle) || !(vMatrix = mHandle.get_rep()) ) {
    error( "No handle or representation" );
    return;
  }

  // Check to see if the input data has changed.
  if( generation_ != mHandle->generation ) {
    generation_  = mHandle->generation;

    update = true;
  }

  if( !mHandle_.get_rep() || update ) {

    // Get the size of the matrix.
    int nRows = vMatrix->nrows();
    int nCols = vMatrix->ncols();

    double mag;

    ColumnMatrix *mMatrix = scinew ColumnMatrix(nRows);

    for( int i=0; i<nRows; i++ ) {
      mag = 0;

      for( int j=0; j<nCols; j++ )
	mag += ( vMatrix->get(i, j) * vMatrix->get(i, j) );

      mag = sqrt( mag );

      mMatrix->put(i, 0, mag);
    }

    mHandle_ = MatrixHandle( mMatrix );
  }

  // Get a handle to the output  matrix port.
  MatrixOPort *omatrix_port =
    (MatrixOPort *) get_oport("Output Magnitude Scalar");

  if (!omatrix_port) {
    error( "Unable to initialize "+name+"'s oport" );
    return;
  }

  // Send the data downstream
  omatrix_port->send( mHandle_ );
}

void VectorMagnitude::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Fusion
