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
 *  MDSPlusFieldReader.cc:
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

#include <Dataflow/Ports/FieldPort.h>

#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/HexVolField.h>
#include <Packages/Fusion/share/share.h>

#include <Packages/Fusion/share/share.h>

#include <Packages/Fusion/Core/ThirdParty/mdsPlusAPI.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE MDSPlusFieldReader : public Module {
public:
  MDSPlusFieldReader(const string& id);

  virtual ~MDSPlusFieldReader();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);

protected:
  GuiString sServerName_;
  GuiString sTreeName_;
  GuiString sShotNumber_;
  GuiString sSliceNumber_;

  GuiInt bPressure_;
  GuiInt bBField_;
  GuiInt bVField_;

  string server_;
  string tree_;
  int shot_;
  int slice_;

  int pressure_;
  int bField_;
  int vField_;

  FieldHandle bHandle_;
  FieldHandle vHandle_;
  FieldHandle pHandle_;
};

extern "C" FusionSHARE Module* make_MDSPlusFieldReader(const string& id) {
  return scinew MDSPlusFieldReader(id);
}

MDSPlusFieldReader::MDSPlusFieldReader(const string& id)
  : Module("MDSPlusFieldReader", id, Source, "DataIO", "Fusion"),
    sServerName_("serverName", id, this),
    sTreeName_("treeName", id, this),
    sShotNumber_("shotNumber", id, this),
    sSliceNumber_("sliceNumber", id, this),

    bPressure_("bPressure", id, this),
    bBField_("bBField", id, this),
    bVField_("bVField", id, this),
    pressure_(-1),
    bField_(-1),
    vField_(-1)
{
}

MDSPlusFieldReader::~MDSPlusFieldReader(){
}

void MDSPlusFieldReader::execute(){

#ifdef MDSPLUS

  string server(sServerName_.get());      // MDS+ Server
  std::string tree(sTreeName_.get());     // MDS+ Tree 
  int shot = atoi( sShotNumber_.get().c_str() );  // NIMROD shot to be opened
  int slice =  atoi( sSliceNumber_.get().c_str() );

  bool pressure = bPressure_.get();
  bool bField = bBField_.get();
  bool vField = bVField_.get();

  if( server_ != server ||
      tree_   != tree   ||
      shot_   != shot   ||
      slice_  != slice  ||

      pressure_ != pressure ||
      bField_ != bField ||
      vField  != vField ) {

    server_ = server;
    tree_   = tree;
    shot_   = shot;
    slice_  = slice;

    pressure_ = pressure;
    bField_ = bField;
    vField_ = vField;

    std::string sNode("SCALARS");                 // Node/Data to fetch
    std::string vNode("VECTORS.CYLINDRICAL");     // Node/Data to fetch

    int nRadial, nTheta, nPhi;  // Dimension of the data.    
   
    char *name = NULL;          // Used to hold the name of the slice 
    double time;                // Used to hold the time of the slice 
    int nSlices;                // Number of time slices.
    int *nids;                  // IDs to the time slice nodes in the tree 

    double     *grid_data[3];   // Grid data.
    double  *b_field_data[3];   // Time slice B Field  data. 
    double  *v_field_data[3];   // Time slice V Field  data. 
    double *pressure_data;      // Time slice Pressure data. 

    pressure_data = NULL;

    for( int i=0; i<3; i++ )
      grid_data[i] = b_field_data[i] = v_field_data[i] = NULL;

    std::string axis;

    /* Connect to MDSplus */
    if( MDS_Connect(server.c_str()) < 0 ) {
      error( "Error connecting to Mds Server " + server );
      return;
    }
    else
      cout << "MDSPLUSFieldReader - Conecting to MdsPlus Server --> " << server << endl;

    // Open tree
    if( MDS_Open( tree.c_str(), shot) , 0 ) {
      //	    error( "Error opening " + tree + " tree for shot " + shot );
      return;
    }
    else
      cout << "MDSPLUSFieldReader - Opening " << tree << "tree for shot " << shot << endl;

    int dims[3];

    // Query the server for the cylindrical components of the grid.
    for( int i=0; i<3; i++ ) {
      if( i == 0 )
	axis = "R";
      else if( i == 1 )
	axis = "Z";
      else if( i == 2 )
	axis = "PHI";

      grid_data[i] = get_grid( axis.c_str(), dims );

      if( i == 0 ) {
	nRadial = dims[0]; nTheta = dims[1]; }
      else if( i == 1 ) {
	if( nRadial != dims[0] || nTheta != dims[1] )
	{
	  error( "Error Grid dims do not match: " );
	  //			dims[0] + " != " + nRadial + "  " +
	  //			dims[1] + " != " + nTheta );

	  return;
	}
      }
      else if( i == 2 )
	nPhi = dims[0];
    }

    cout << "MDSPLUSFieldReader - Grid size: " << nRadial << " " << nTheta << " " << nPhi << endl;

    std::string buf;

    // Query the server for the number of slices in the tree
    nSlices = get_slice_ids( &nids );

    cout << "MDSPLUSFieldReader - Number of slices in the current shot " << nSlices << endl;

    if( 0<=slice && slice<nSlices ) {

      name = get_slice_name( nids, slice );
      time = get_slice_time( name );

      cout << "MDSPLUSFieldReader - Processing slice " << name << " at time " << time << endl;

      // Fetch the Pressure data from the node
      if( pressure_ ) {
	buf = sNode + ":PRESSURE";
	pressure_data = get_realspace_data( name, buf.c_str(), dims );
      }

      // Fetch the B Field data from the node
      if( bField_ ) {
	buf = vNode + ":BFIELD:R";
	b_field_data[0] = get_realspace_data( name, buf.c_str(), dims );

	buf = vNode + ":BFIELD:Z";
	b_field_data[1] = get_realspace_data( name, buf.c_str(), dims );

	buf = vNode + ":BFIELD:PHI";
	b_field_data[2] = get_realspace_data( name, buf.c_str(), dims );
      }

      // Fetch the Velocity Field data from the node
      if( vField_ ) {
	buf = vNode + ":VFIELD:R";
	v_field_data[0] = get_realspace_data( name, buf.c_str(), dims );

	buf = vNode + ":VFIELD:Z";
	v_field_data[1] = get_realspace_data( name, buf.c_str(), dims );

	buf = vNode + ":VFIELD:PHI";
	v_field_data[2] = get_realspace_data( name, buf.c_str(), dims );
      }

      int idim = nRadial;
      int jdim = nTheta;
      int kdim = nPhi;

      // Create the grid, and scalar and vector data matrices.
      HexVolMesh *hvm = scinew HexVolMesh;

      Vector *bVector = NULL;
      Vector *vVector = NULL;
      double *pValues = NULL;


      if( bField_ )
	bVector = scinew Vector[idim*jdim*kdim];

      if( vField_ )
	vVector = scinew Vector[idim*jdim*kdim];

      if( pressure_ )
	pValues = scinew double[idim*jdim*kdim];

      cout << "MDSPLUSFieldReader - Storing data. " << endl;

      // Read the data.
      double xVal, yVal, zVal, pVal, rad, phi;

      int cc = 0;

      for( int k=0; k<kdim; k++ ) {

	phi = grid_data[2][k];

	for( int j=0; j<jdim; j++ ) {
	  for( int i=0; i<idim; i++ ) {

	    rad = grid_data[0][i+j*idim];

	    xVal =  rad * cos( phi );
	    yVal = -rad * sin( phi );
	    zVal =  grid_data[1][i+j*idim];

	    hvm->add_point( Point( xVal, yVal, zVal ) );

	    if( bField_ )
	    {
	      xVal =  b_field_data[0][cc] * cos(phi) -
		b_field_data[2][cc] * sin(phi);
	      yVal = -b_field_data[0][cc] * sin(phi) -
		b_field_data[2][cc] * cos(phi );

	      zVal =  b_field_data[1][cc];

	      bVector[cc] = Vector( xVal, yVal, zVal);
	    }

	    if( vField_ )
	    {
	      xVal =  v_field_data[0][cc] * cos(phi) -
		v_field_data[2][cc] * sin(phi);
	      yVal = -v_field_data[0][cc] * sin(phi) -
		v_field_data[2][cc] * cos(phi );

	      zVal =  v_field_data[1][cc];

	      vVector[cc] = Vector( xVal, yVal, zVal);
	    }

	    if( pressure_ ) {
	      pVal = pressure_data[cc];

	      pValues[cc] = pVal;
	    }

	    cc++;
	  }
	}
      }

      cout << "MDSPLUSFieldReader - Creating mesh. " << endl;

      // Create the Hex Vol Mesh.
      HexVolMesh::Node::array_type nnodes(8);

      int ijdim = idim * jdim;

      int istart=0, iend=idim;
      int jstart=0, jend=jdim;
      int kstart=0, kend=kdim/2;

      int iskip = 10;
      int jskip = 5;

      int i,  j,  k;
      int i0, j0, k0;
      int i1, j1, k1;

      for( k = kstart; k<kend; k++ ) { 

	k0 = (k    ) % kdim;
	k1 = (k + 1) % kdim;

	for( j = jstart; j<jend; j+=jskip ) {
 
	  j0 = (j        ) % jdim;
	  j1 = (j + jskip) % jdim;

	  if( j1 > jend - 1)
	    j1 = jend - 1;

	  for( i = istart; i<iend-1; i+=iskip ) { 

	    i0 = i;
	    i1 = i + iskip;

	    if( i1 > iend - 1)
	      i1 = iend - 1;

	    nnodes[0] = i0 + j0 * idim + k0 * ijdim;
	    nnodes[1] = i1 + j0 * idim + k0 * ijdim;
	    nnodes[2] = i1 + j1 * idim + k0 * ijdim;
	    nnodes[3] = i0 + j1 * idim + k0 * ijdim;   
	    nnodes[4] = i0 + j0 * idim + k1 * ijdim;
	    nnodes[5] = i1 + j0 * idim + k1 * ijdim;
	    nnodes[6] = i1 + j1 * idim + k1 * ijdim;
	    nnodes[7] = i0 + j1 * idim + k1 * ijdim;   

	    hvm->add_elem(nnodes);
	  }
	}
      }


      hvm->set_property( "I Dim", idim, false );
      hvm->set_property( "J Dim", jdim, false );
      hvm->set_property( "K Dim", kdim, false );

      int ijkdim = idim * jdim * kdim;

      // Now after the mesh has been created, create the field and put the
      // data into the field.

      if( pValues ) {

	HexVolField<double> *hvfP =
	  scinew HexVolField<double>(HexVolMeshHandle(hvm), Field::NODE);

	pHandle_ = FieldHandle( hvfP );

	// Add the data to the field.
	HexVolField<double>::fdata_type::iterator outP = hvfP->fdata().begin();

	for( int i=0; i<ijkdim; i++ ) {

	  *outP = pValues[i];
	  outP++;
	}
      }


      if( bVector ) {
	HexVolField<Vector> *hvfB =
	  scinew HexVolField<Vector>(HexVolMeshHandle(hvm), Field::NODE);

	bHandle_ = FieldHandle( hvfB );

	// Add the data to the field.
	HexVolField<Vector>::fdata_type::iterator outB = hvfB->fdata().begin();

	for( int i=0; i<ijkdim; i++ ) {
	  *outB = bVector[i];
	  outB++;
	}
      }


      if( vVector ) {
	HexVolField<Vector> *hvfV =
	  scinew HexVolField<Vector>(HexVolMeshHandle(hvm), Field::NODE);
    
	vHandle_ = FieldHandle( hvfV );

	// Add the data to the field.
	HexVolField<Vector>::fdata_type::iterator outV = hvfV->fdata().begin();

	for( int i=0; i<ijkdim; i++ ) {
	  *outV = vVector[i];
	  outV++;
	}
      }
    }
    else {
      error( "Not a valid slice." );
      return;
    }
  }

  // Get a handle to the output pressure field port.
  if( pHandle_.get_rep() )
  {
    FieldOPort *ofield_port =
      (FieldOPort *) get_oport("Output Pressure Field");

    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( pHandle_ );
  }

  // Get a handle to the output B Field field port.
  if( bHandle_.get_rep() )
  {
    FieldOPort *ofield_port =
      (FieldOPort *)get_oport("Output B Field");

    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( bHandle_ );
  }

  // Get a handle to the output V Field field port.
  if( vHandle_.get_rep() )
  {
    FieldOPort *ofield_port =
      (FieldOPort *) get_oport("Output V Field");

    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( vHandle_ );
  }

#else

  error( "No MDS PLUS availible." );

#endif
}

void MDSPlusFieldReader::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Fusion
