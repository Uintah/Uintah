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
 *  MeshBuilde.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   March 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/GuiInterface/GuiVar.h>

#include <Dataflow/Ports/FieldPort.h>

#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/HexVolField.h>
#include <Packages/Fusion/share/share.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE EditFusionField : public Module {
public:
  EditFusionField(const string& id);

  virtual ~EditFusionField();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);

private:
  GuiInt iDim_;
  GuiInt jDim_;
  GuiInt kDim_;

  GuiInt iStart_;
  GuiInt jStart_;
  GuiInt kStart_;

  GuiInt iDelta_;
  GuiInt jDelta_;
  GuiInt kDelta_;

  GuiInt iSkip_;
  GuiInt jSkip_;
  GuiInt kSkip_;

  GuiInt reducePts_;

  int idim_;
  int jdim_;
  int kdim_;

  int istart_;
  int jstart_;
  int kstart_;

  int iend_;
  int jend_;
  int kend_;

  int iskip_;
  int jskip_;
  int kskip_;

  int reduce_;

  FieldHandle  fHandle_;

  int fGeneration_;
};

extern "C" FusionSHARE Module* make_EditFusionField(const string& id) {
  return scinew EditFusionField(id);
}

EditFusionField::EditFusionField(const string& id)
  : Module("EditFusionField", id, Source, "Fields", "Fusion"),

    iDim_("idim", id, this),
    jDim_("jdim", id, this),
    kDim_("kdim", id, this),

    iStart_("istart", id, this),
    jStart_("jstart", id, this),
    kStart_("kstart", id, this),

    iDelta_("idelta", id, this),
    jDelta_("jdelta", id, this),
    kDelta_("kdelta", id, this),

    iSkip_("iskip", id, this),
    jSkip_("jskip", id, this),
    kSkip_("kskip", id, this),

    reducePts_("reduce", id, this),

    idim_(0),
    jdim_(0),
    kdim_(0),

    istart_(-1),
    jstart_(-1),
    kstart_(-1),

    iend_(-1),
    jend_(-1),
    kend_(-1),

    iskip_(10),
    jskip_(5),
    kskip_(1),

    reduce_( false ),

    fGeneration_(-1)
{
}

EditFusionField::~EditFusionField(){
}

void EditFusionField::execute(){

  bool updateAll    = false;
  bool updateField  = false;

  FieldHandle fHandle;

  Field *hvfInput;
  HexVolMesh *hvmInput;

  // Get a handle to the input field port.
  FieldIPort* ifield_port = (FieldIPort *) get_iport("Input Field");

  if (!ifield_port) {
    error( "Unable to initialize "+name+"'s iport" );
    return;
  }

  // The field input is required.
  if (!ifield_port->get(fHandle) ||
      !(hvfInput = fHandle.get_rep()) ||
      !(hvmInput = (HexVolMesh*) fHandle->mesh().get_rep())) {
    error( "No handle or representation" );
    return;
  }


  // Check to see if the input field has changed.
  if( fGeneration_ != fHandle->generation ) {
    fGeneration_ = fHandle->generation;

    cout << "EditFusionField - New Field Data." << endl;

    updateField = true;
  }


  if( !hvmInput->get_property( "I Dim", idim_ ) ||
      !hvmInput->get_property( "J Dim", jdim_ ) ||
      !hvmInput->get_property( "K Dim", kdim_ ) ) {
    error( "No Dimensions." );
    return;
  }

  HexVolMesh::Node::size_type npts;

  hvmInput->size( npts );

  // Make sure they match the number of points in the mesh.
  if( npts != idim_ * jdim_ * kdim_ ) {
    error( "Mesh dimensions do not match mesh size." );
    return;
  }


  if( reduce_ != reducePts_.get() ) {
    reduce_  = reducePts_.get();

    updateAll = true;
  }


  // Check to see if the dimensions have changed.
  if( idim_ != iDim_.get() ||
      jdim_ != jDim_.get() ||
      kdim_ != kDim_.get() ) {

    cout << "EditFusionField - Updating GUI." << endl;

    // Update the dims in the GUI.
    ostringstream str;
    str << id << " set_size " << idim_ << " " << jdim_ << " " << kdim_;

    TCL::execute(str.str().c_str());

    updateAll = true;
  }


  // Check to see if the user setable values have changed.
  if( istart_ != iStart_.get() ||
      jstart_ != jStart_.get() ||
      kstart_ != kStart_.get() ||

      iend_  != (iStart_.get() + iDelta_.get()) ||
      jend_  != (jStart_.get() + jDelta_.get()) ||
      kend_  != (kStart_.get() + kDelta_.get()) ||

      iskip_ != iSkip_.get()  ||
      jskip_ != jSkip_.get() ||
      kskip_ != kSkip_.get() ) {

    cout << "EditFusionField - Updating values." << endl;

    istart_ = iStart_.get();
    jstart_ = jStart_.get();
    kstart_ = kStart_.get();

    iend_ = iStart_.get() + iDelta_.get();
    jend_ = jStart_.get() + jDelta_.get();
    kend_ = kStart_.get() + kDelta_.get();

    iskip_ = iSkip_.get();
    jskip_ = jSkip_.get();
    kskip_ = kSkip_.get();

    updateAll = true;
  }

  // If no data or a changed recreate the mesh.
  if( !fHandle_.get_rep() || updateAll || updateField ) {

    HexVolMesh *hvm = scinew HexVolMesh;

    //  Only store the nodes being used in the mesh.
    if( reduce_ ) {

      // Create the new Hex Vol Mesh.
      HexVolMesh::Node::array_type nnodes(8);

      int ijdim = idim_ * jdim_;

      int i,  j,  k;
      int i0, j0, k0;
      int i1, j1, k1;

      int iend=0, jend=0, kend=0;

      int index;

      // First reduce the nodes down.
      for( k=kstart_; k<kend_+kskip_; k+=kskip_ ) { 

	k0 = k % kdim_;

	// Check for overlap.
	if( k-kskip_ <= kstart_+kdim_ && kstart_+kdim_ <= k )
	  k0 = kstart_;

	kend++;

	jend = 0;

	for( j=jstart_; j<jend_+jskip_; j+=jskip_ ) {
 
	  j0 = j % jdim_;

	  // Check for overlap.
	  if( j-jskip_ <= jstart_+jdim_ && jstart_+jdim_ <= j )
	    j0 = jstart_;

	  jend++;

	  iend = 0;

	  for( i=istart_; i<iend_+iskip_; i+=iskip_ ) { 

	    i0 = i;

	    if( i0 > idim_ - 1 )
	      i0 = idim_ - 1;

	    iend++;

	    index = i0 + j0 * idim_ + k0 * ijdim;

	    hvm->add_point( hvmInput->point( index ) );
	  }
	}
      }

      ijdim = iend * jend;

      // Create the new mesh based on the reduced points.
      for( k=0; k<kend-1; k++ ) { 

	k0 = (k    ) % kend;
	k1 = (k + 1) % kend;

	for( j=0; j<jend-1; j++ ) {
 
	  j0 = (j    ) % jend;
	  j1 = (j + 1) % jend;

	  for( i=0; i<iend-1; i++ ) { 

	    i0 = i;
	    i1 = i + 1;

	    nnodes[0] = i0 + j0 * iend + k0 * ijdim;
	    nnodes[1] = i1 + j0 * iend + k0 * ijdim;
	    nnodes[2] = i1 + j1 * iend + k0 * ijdim;
	    nnodes[3] = i0 + j1 * iend + k0 * ijdim;   
	    nnodes[4] = i0 + j0 * iend + k1 * ijdim;
	    nnodes[5] = i1 + j0 * iend + k1 * ijdim;
	    nnodes[6] = i1 + j1 * iend + k1 * ijdim;
	    nnodes[7] = i0 + j1 * iend + k1 * ijdim;   

	    hvm->add_elem(nnodes);
	  }
	}
      }

      hvm->set_property( "I Dim", iend, false );
      hvm->set_property( "J Dim", jend, false );
      hvm->set_property( "K Dim", kend, false );


      // Now after the mesh has been created, create the field and put the
      // data into the field.

      HexVolField<double> *hvfD = NULL;
      HexVolField<Vector> *hvfV = NULL;

      // Editing a Scalar HexVolField.
      if( (hvfD = dynamic_cast<HexVolField<double> *>(hvfInput)) ) {
	HexVolField<double> *hvf =
	  scinew HexVolField<double>(HexVolMeshHandle(hvm), Field::NODE);

	fHandle_ = FieldHandle( hvf );
	HexVolField<double>::fdata_type::iterator out = hvf->fdata().begin();

	ijdim = idim_ * jdim_;
	iend=0; jend=0; kend=0;

	// First reduce the data down.
	for( k=kstart_; k<kend_+kskip_; k+=kskip_ ) { 

	  k0 = k % kdim_;

	  // Check for overlap.
	  if( k-kskip_ <= kstart_+kdim_ && kstart_+kdim_ <= k )
	    k0 = kstart_;

	  for( j=jstart_; j<jend_+jskip_; j+=jskip_ ) {
 
	    j0 = j % jdim_;

	    // Check for overlap.
	    if( j-jskip_ <= jstart_+jdim_ && jstart_+jdim_ <= j )
	      j0 = jstart_;

	    for( i=istart_; i<iend_+iskip_; i+=iskip_ ) { 

	      i0 = i;

	      if( i0 > idim_ - 1 )
		i0 = idim_ - 1;

	      index = i0 + j0 * idim_ + k0 * ijdim;

	      *out = hvfD->fdata()[index];
	      out++;
	    }
	  }
	}
      }

      // Editing a Vector HexVolField.
      else if( (hvfV = dynamic_cast<HexVolField<Vector> *>(hvfInput)) ) {
	HexVolField<Vector> *hvf =
	  scinew HexVolField<Vector>(HexVolMeshHandle(hvm), Field::NODE);

	fHandle_ = FieldHandle( hvf );
	HexVolField<Vector>::fdata_type::iterator out = hvf->fdata().begin();

	ijdim = idim_ * jdim_;

	// First reduce the data down.
	for( k=kstart_; k<kend_+kskip_; k+=kskip_ ) { 

	  k0 = k % kdim_;

	  // Check for overlap.
	  if( k-kskip_ <= kstart_+kdim_ && kstart_+kdim_ <= k )
	    k0 = kstart_;

	  for( j=jstart_; j<jend_+jskip_; j+=jskip_ ) {
 
	    j0 = j % jdim_;

	    // Check for overlap.
	    if( j-jskip_ <= jstart_+jdim_ && jstart_+jdim_ <= j )
	      j0 = jstart_;

	    for( i=istart_; i<iend_+iskip_; i+=iskip_ ) { 

	      i0 = i;

	      if( i0 > idim_ - 1 )
		i0 = idim_ - 1;

	      index = i0 + j0 * idim_ + k0 * ijdim;

	      *out = hvfV->fdata()[index];
	      out++;
	    }
	  }
	}
      }
      else {
	error( "Only availible for HexVol Scalar and Vector data" );
	return;
      }

      cout << "EditFusionField - Reducing " << iend << " " << jend << " " << kend << endl;
    }
    else
    {
      // transfer all the points from the old mesh to the new mesh.
      for( int i=0; i<npts; i++ )
	hvm->add_point( hvmInput->point( i ) );

      // Create the new Hex Vol Mesh.
      HexVolMesh::Node::array_type nnodes(8);

      int ijdim = idim_ * jdim_;

      int i,  j,  k;
      int i0, j0, k0;
      int i1, j1, k1;

      for( k=kstart_; k<kend_; k+=kskip_ ) { 

	k0 = (k         ) % kdim_;
	k1 = (k + kskip_) % kdim_;

	// Check for overlap.
	if( k <= kstart_+kdim_ && kstart_+kdim_ <= k+kskip_ )
	  k1 = kstart_;

	for( j=jstart_; j<jend_; j+=jskip_ ) {
 
	  j0 = (j         ) % jdim_;
	  j1 = (j + jskip_) % jdim_;

	  // Check for overlap.
	  if( j <= jstart_+jdim_ && jstart_+jdim_ <= j+jskip_ )
	    j1 = jstart_;

	  for( i=istart_; i<iend_; i+=iskip_ ) { 

	    i0 = i;
	    i1 = i + iskip_;

	    if( i1 > idim_ - 1 )
	      i1 = idim_ - 1;

	    if( i0 < idim_ - 1 && i0 != i1 ) {

	      nnodes[0] = i0 + j0 * idim_ + k0 * ijdim;
	      nnodes[1] = i1 + j0 * idim_ + k0 * ijdim;
	      nnodes[2] = i1 + j1 * idim_ + k0 * ijdim;
	      nnodes[3] = i0 + j1 * idim_ + k0 * ijdim;   
	      nnodes[4] = i0 + j0 * idim_ + k1 * ijdim;
	      nnodes[5] = i1 + j0 * idim_ + k1 * ijdim;
	      nnodes[6] = i1 + j1 * idim_ + k1 * ijdim;
	      nnodes[7] = i0 + j1 * idim_ + k1 * ijdim;   

	      hvm->add_elem(nnodes);
	    }
	  }
	}
      }

      hvm->set_property( "I Dim", idim_, false );
      hvm->set_property( "J Dim", jdim_, false );
      hvm->set_property( "K Dim", kdim_, false );

      // Now after the mesh has been created, create the field.
    
      HexVolField<Vector> *hvfV = NULL;
      HexVolField<double> *hvfD = NULL;

      if( (hvfV = dynamic_cast<HexVolField<Vector> *>(hvfInput)) ) {
	HexVolField<Vector> *hvf =
	  scinew HexVolField<Vector>(HexVolMeshHandle(hvm), Field::NODE);

	fHandle_ = FieldHandle( hvf );

	HexVolField<Vector>::fdata_type::iterator  in = hvfV->fdata().begin();
	HexVolField<Vector>::fdata_type::iterator out = hvf->fdata().begin();
	HexVolField<Vector>::fdata_type::iterator end = hvfV->fdata().end();

	while (in != end) {
	  *out = *in;
	  ++in; ++out;
	}
      }

      else if( (hvfD = dynamic_cast<HexVolField<double> *>(hvfInput)) ) {
	HexVolField<double> *hvf =
	  scinew HexVolField<double>(HexVolMeshHandle(hvm), Field::NODE);

	fHandle_ = FieldHandle( hvf );

	HexVolField<double>::fdata_type::iterator  in = hvfD->fdata().begin();
	HexVolField<double>::fdata_type::iterator out = hvf->fdata().begin();
	HexVolField<double>::fdata_type::iterator end = hvfD->fdata().end();

	while (in != end) {
	  *out = *in;
	  ++in; ++out;
	}
      }
    }
  }

  // Get a handle to the output field port.
  if( fHandle_.get_rep() )
  {
    FieldOPort *ofield_port =
      (FieldOPort *) get_oport("Output Field");

    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( fHandle_ );
  }
}

void EditFusionField::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Fusion
