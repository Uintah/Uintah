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

#include <sci_defs.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Ports/FieldPort.h>

#include <Packages/Fusion/Core/Datatypes/StructHexVolField.h>
#include <Packages/Fusion/share/share.h>

#include <Packages/Fusion/Core/ThirdParty/mdsPlusAPI.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE MDSPlusFieldReader : public Module {
public:
  MDSPlusFieldReader(GuiContext *context);

  virtual ~MDSPlusFieldReader();

  virtual void execute();

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


DECLARE_MAKER(MDSPlusFieldReader)


MDSPlusFieldReader::MDSPlusFieldReader(GuiContext *context)
  : Module("MDSPlusFieldReader", context, Source, "DataIO", "Fusion"),
    sServerName_(context->subVar("serverName")),
    sTreeName_(context->subVar("treeName")),
    sShotNumber_(context->subVar("shotNumber")),
    sSliceNumber_(context->subVar("sliceNumber")),
    bPressure_(context->subVar("bPressure")),
    bBField_(context->subVar("bBField")),
    bVField_(context->subVar("bVField")),
    pressure_(-1),
    bField_(-1),
    vField_(-1)
{
}

MDSPlusFieldReader::~MDSPlusFieldReader(){
}

void MDSPlusFieldReader::execute(){

#ifdef HAVE_MDSPLUS

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
    std::string axis;

    int nRadial, nTheta, nPhi;  // Dimension of the data.    
   
    char *name = NULL;          // Used to hold the name of the slice 
    double time;                // Used to hold the time of the slice 
    int nSlices;                // Number of time slices.
    int *nids;                  // IDs to the time slice nodes in the tree 

    double     *grid_data[3];   // Grid data.
    double  *b_field_data[3];   // Time slice B Field  data. 
    double  *v_field_data[3];   // Time slice V Field  data. 
    double *pressure_data;      // Time slice Pressure data. 

    for( int i=0; i<3; i++ )
      grid_data[i] = b_field_data[i] = v_field_data[i] = NULL;

    pressure_data = NULL;


    StructHexVolMesh *hvm = NULL;
    StructHexVolField<Vector> *bfield = NULL;
    StructHexVolField<Vector> *vfield = NULL;
    StructHexVolField<double> *pfield = NULL;

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

      
      hvm = scinew StructHexVolMesh(nRadial, nTheta, nPhi);

      // Fetch the Pressure data from the node
      if( pressure_ ) {
	buf = sNode + ":PRESSURE";
	pressure_data = get_realspace_data( name, buf.c_str(), dims );

	pfield = scinew StructHexVolField<double>(hvm, Field::NODE);
	pHandle_ = pfield;
      }

      // Fetch the B Field data from the node
      if( bField_ ) {
	buf = vNode + ":BFIELD:R";
	b_field_data[0] = get_realspace_data( name, buf.c_str(), dims );

	buf = vNode + ":BFIELD:Z";
	b_field_data[1] = get_realspace_data( name, buf.c_str(), dims );

	buf = vNode + ":BFIELD:PHI";
	b_field_data[2] = get_realspace_data( name, buf.c_str(), dims );

	bfield = scinew StructHexVolField<Vector>(hvm, Field::NODE);
	bHandle_ = bfield;
      }

      // Fetch the Velocity Field data from the node
      if( vField_ ) {
	buf = vNode + ":VFIELD:R";
	v_field_data[0] = get_realspace_data( name, buf.c_str(), dims );

	buf = vNode + ":VFIELD:Z";
	v_field_data[1] = get_realspace_data( name, buf.c_str(), dims );

	buf = vNode + ":VFIELD:PHI";
	v_field_data[2] = get_realspace_data( name, buf.c_str(), dims );

	vfield = scinew StructHexVolField<Vector>(hvm, Field::NODE);
	vHandle_ = vfield;
      }

      unsigned int idim = nRadial;
      unsigned int jdim = nTheta;
      unsigned int kdim = nPhi;

      cout << "MDSPLUSFieldReader - Creating mesh and field. " << endl;

      // Convert the data and place in the mesh and field.
      double xVal, yVal, zVal, pVal, rad, phi;

      StructHexVolMesh::Node::index_type node;

      register unsigned int i, j, k, cc = 0;

      for( k=0; k<kdim; k++ ) {

	phi = grid_data[2][k];

	for( j=0; j<jdim; j++ ) {
	  for( i=0; i<idim; i++ ) {

	    node.i_ = i;
	    node.j_ = j;
	    node.k_ = k;

	    rad = grid_data[0][i+j*idim];

	    xVal =  rad * cos( phi );
	    yVal = -rad * sin( phi );
	    zVal =  grid_data[1][i+j*idim];

	    hvm->set_point(node, Point( xVal, yVal, zVal ) );

	    if( bField_ )
	    {
	      xVal =  b_field_data[0][cc] * cos(phi) -
		b_field_data[2][cc] * sin(phi);
	      yVal = -b_field_data[0][cc] * sin(phi) -
		b_field_data[2][cc] * cos(phi );

	      zVal =  b_field_data[1][cc];

	      bfield->set_value(Vector(xVal, yVal, zVal), node);
	    }

	    if( vField_ )
	    {
	      xVal =  v_field_data[0][cc] * cos(phi) -
		v_field_data[2][cc] * sin(phi);
	      yVal = -v_field_data[0][cc] * sin(phi) -
		v_field_data[2][cc] * cos(phi );

	      zVal =  v_field_data[1][cc];

	      vfield->set_value(Vector(xVal, yVal, zVal), node);
	    }

	    if( pressure_ ) {
	      pVal = pressure_data[cc];

	      pfield->set_value(pVal, node);
	    }

	    cc++;
	  }
	}
      }
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


} // End namespace Fusion
