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

#define MAX_GRID 4
#define MAX_SCALAR 1
#define MAX_VECTOR 3

using namespace SCIRun;

class FusionSHARE MDSPlusFieldReader : public Module {
public:
  MDSPlusFieldReader(GuiContext *context);

  virtual ~MDSPlusFieldReader();

  virtual void execute();

protected:
  enum { REALSPACE=0, PERTURBED=1 };

  GuiString sServerName_;
  GuiString sTreeName_;
  GuiString sShotNumber_;
  GuiString sSliceNumber_;

  GuiInt iSliceRange_;
  GuiString sSliceStart_;
  GuiString sSliceStop_;

  GuiInt iScalar0_;
  GuiInt iVector0_;
  GuiInt iVector1_;
  GuiInt iVector2_;

  GuiInt iSpace_;
  GuiInt iMode_;

  string server_;
  string tree_;
  int shot_;
  int slice_;
  int sliceRange_;
  int sliceStart_;
  int sliceStop_;

  int bScalar_[MAX_SCALAR];
  int bVector_[MAX_VECTOR];

  int space_;
  int mode_;

  bool error_;

  FieldHandle sHandle_[4];
  FieldHandle vHandle_[3];

  double    *grid_data[MAX_GRID];           // Grid data - R Z PHI and K.
  double  *scalar_data[MAX_SCALAR][3];      // Time slice scalar data. 
  double  *vector_data[MAX_VECTOR][3][3];   // Time slice vector data. 

  int nRadial, nTheta, nPhi, nMode;  // Dimension of the data.    
};


DECLARE_MAKER(MDSPlusFieldReader)


MDSPlusFieldReader::MDSPlusFieldReader(GuiContext *context)
  : Module("MDSPlusFieldReader", context, Source, "DataIO", "Fusion"),
    sServerName_(context->subVar("serverName")),
    sTreeName_(context->subVar("treeName")),
    sShotNumber_(context->subVar("shotNumber")),
    sSliceNumber_(context->subVar("sliceNumber")),
    iSliceRange_(context->subVar("sliceRange")),
    sSliceStart_(context->subVar("sliceStart")),
    sSliceStop_(context->subVar("sliceStop")),
    iScalar0_(context->subVar("bPressure")),
    iVector0_(context->subVar("bBField")),
    iVector1_(context->subVar("bVField")),
    iVector2_(context->subVar("bJField")),
    iSpace_(context->subVar("space")),
    iMode_(context->subVar("mode")),
    space_(-1),
    mode_(-1),
    error_(false)
{

  for( int i=0; i<MAX_SCALAR; i++ )
    bScalar_[i] = -1;

  for( int i=0; i<MAX_VECTOR; i++ )
    bVector_[i] = -1;
}

MDSPlusFieldReader::~MDSPlusFieldReader(){
}

void MDSPlusFieldReader::execute(){

#ifdef HAVE_MDSPLUS

  string sPortStr[MAX_SCALAR] = {"Output Pressure Field" };
  string vPortStr[MAX_VECTOR] = {"Output B Field", "Output V Field", "Output J Field", };
  string gridStr[MAX_GRID] = {"R", "Z", "PHI", "K" };
  string spaceStr[3] = {"REALSPACE", "PERTURBED.REAL", "PERTURBED.IMAG" };

  string sFieldStr[MAX_SCALAR] =
    {":PRESSURE"};

  string vFieldStr[MAX_VECTOR][3] =
    { {":BFIELD:R", ":BFIELD:Z", ":BFIELD:PHI"},
      {":VFIELD:R", ":VFIELD:Z", ":VFIELD:PHI"},
      {":JFIELD:R", ":JFIELD:Z", ":JFIELD:PHI"} };

  string server(sServerName_.get());      // MDS+ Server
  std::string tree(sTreeName_.get());     // MDS+ Tree 
  int shot = atoi( sShotNumber_.get().c_str() );  // NIMROD shot to be opened
  int slice =  atoi( sSliceNumber_.get().c_str() );
  int sliceRange =  iSliceRange_.get();
  int sliceStart =  atoi( sSliceStart_.get().c_str() );
  int sliceStop =  atoi( sSliceStop_.get().c_str() );

  bool update = false;

  if( bScalar_[0] != iScalar0_.get() ) {
    bScalar_[0] = iScalar0_.get();
    update = true;
  }

  if( bVector_[0] != iVector0_.get() ) {
    bVector_[0] = iVector0_.get();
    update = true;
  }

  if( bVector_[1] != iVector1_.get() ) {
    bVector_[1] = iVector1_.get();
    update = true;
  }

  if( bVector_[2] != iVector2_.get() ) {
    bVector_[2] = iVector2_.get();
    update = true;
  }

  int space = iSpace_.get();
  int mode = iMode_.get();

  bool readAll, readGrid, readData;
  bool modeChange;

  if( error_  == true ||
      
      update == true ||

      server_ != server ||
      tree_   != tree   ||
      shot_   != shot ) {

    error_ = false;

    server_ = server;
    tree_   = tree;
    shot_   = shot;

    readAll = readGrid = true;
  } else
    readAll = readGrid = false;

  if( sliceRange_  != sliceRange ||
      sliceStart_ != sliceStart ||
      sliceStop_ != sliceStop ||
      slice_ != slice ||
      space_ != space ) {

    sliceRange_ = sliceRange;
    sliceStart_ = sliceStart;
    sliceStop_ = sliceStop;

    slice_  = slice;
    space_ = space;

    readData = true;
  } else if( sliceRange_ )
    readData = true;
  else
    readData = false;
    
  if( mode_ != mode ) {
    mode_ = mode;
    modeChange = true;
  } else
    modeChange = false;

  if( readAll || readGrid || readData ) {
   
    for( int n=0; n<MAX_GRID; n++ )
      grid_data[n] = NULL;

    for( int n=0; n<MAX_SCALAR; n++ )
      for( int i=0; i<3; i++ )
	scalar_data[n][i] = NULL;

    for( int n=0; n<MAX_VECTOR; n++ )
      for( int i=0; i<3; i++ )
	for( int j=0; j<3; j++ )
	  vector_data[n][i][j] = NULL;

    /* Connect to MDSplus */
    if( MDS_Connect(server.c_str()) < 0 ) {
      error( "Error connecting to Mds Server " + server );
      error_ = true;
      return;
    }
    else
      remark( "Conecting to MdsPlus Server --> " + server );

    // Open tree
    if( MDS_Open( tree.c_str(), shot) , 0 ) {
      ostringstream str;
      str << "Error opening " << tree << " tree for shot " << shot;
      error( str.str() );
      error_ = true;
      return;
    }
    else {
      ostringstream str;
      str << "Opening " << tree << " tree for shot " << shot;
      remark( str.str() );
    }
  }


  if( readAll || readGrid  || readData ) {

    int dims[3];

    // Query the server for the cylindrical components of the grid.
    for( int n=0; n<MAX_GRID; n++ ) {

      dims[0] = dims[1] = dims[2] = 0;

      grid_data[n] = get_grid( gridStr[n].c_str(), dims );

      if( grid_data[n] ) {

	switch( n ) {

	case 0:
	  nRadial = dims[0];
	  nTheta  = dims[1];
	  break;
	
	case 1:
	  if( nRadial != dims[0] || nTheta != dims[1] ) {
	    error( "Error Grid dimensions do not match" );
	    error_ = true;
	    return;
	  } else {
	    break;
	  }

	case 2:
	  nPhi = dims[0];
	  break;
	
	case 3:
	  nMode = dims[0];
	  break;
	}
      } else {
	error( "Error Can not get Grid data" );
	error_ = true;
	return;
      }
    }

    {
      ostringstream str;
      str << "Grid size: " << nRadial << " " << nTheta << " " << nPhi << " " << nMode;
      remark( str.str() );
    }
  }

  if( readAll || readData ) {

    std::string sNode("SCALARS");                 // Node/Data to fetch
    std::string vNode("VECTORS.CYLINDRICAL");     // Node/Data to fetch

    std::string buf;

    int dims[3];

    char *name = NULL;          // Used to hold the name of the slice 
    double time;                // Used to hold the time of the slice 
    int nSlices;                // Number of time slices.
    int *nids;                  // IDs to the time slice nodes in the tree 

    // Query the server for the number of slices in the tree
    nSlices = get_slice_ids( &nids );

    {
      ostringstream str;
      str << "Number of slices in the current shot " << nSlices;
      remark( str.str() );
    }

    
    if( sliceRange_ ) {

      if( sliceStart_< 0 || nSlices<=sliceStop_ ) {
	ostringstream str;
	str << "Slice range is outside of the range [0-" << nSlices << ").";
	error( str.str() );
	error_ = true;
	return;
      } else if( sliceStop_ < sliceStart_ ) {
	ostringstream str;
	str << "Improper slice range";
	error( str.str() );
	error_ = true;
	return;
      }
    } else {

      if( slice_< 0 || nSlices<=slice_ ) {
	ostringstream str;
	str << "Slice " << slice_ << " is outside of the range [0-" << nSlices << ").";
	error( str.str() );
	error_ = true;
	return;
      }

      sliceStart = slice;
      sliceStop  = slice;
    }

    for( slice=sliceStart; slice<=sliceStop; slice++ ) {
      name = get_slice_name( nids, slice );
      time = get_slice_time( name );

      {
	ostringstream str;
	str << "Processing slice " << name << " at time " << time;
	remark( str.str() );
      }


      if( sliceRange_ ) {
	slice_ = slice;

	ostringstream str;
	str << slice_;

	sSliceNumber_.set( str.str().c_str() );
      }

      // Fetch the Scalar data from the node
      for( int n=0; n<MAX_SCALAR; n++ ) {
	if( bScalar_[n] ) {
	
	  buf = sNode + sFieldStr[n];

	  if( space_ == REALSPACE ) {
	    remark( spaceStr[0] + "." + buf );

	    scalar_data[n][0] =
	      get_slice_data( name, spaceStr[0].c_str(), buf.c_str(), dims );

	    if( nRadial != dims[0] || nTheta != dims[1] || nPhi != dims[2] ) {
	      error( "Error dimensions do not match Grid dimensions" );
	      error_ = true;
	      return;
	    }
	  }
	  else if( space_ == PERTURBED ){
	    for( int i=1; i<3; i++ ) { // Real and imaginary perturbed parts
	      remark( spaceStr[i] + "." + buf );

	      scalar_data[n][i] =
		get_slice_data( name, spaceStr[i].c_str(), buf.c_str(), dims );

	      if( nRadial != dims[0] || nTheta != dims[1] || nMode != dims[2] ) {
		error( "Error dimensions do not match Grid dimensions" );
		error_ = true;
		return;
	      }
	    }
	  }
	}
      }

      // Fetch the Vector data from the node
      for( int n=0; n<MAX_VECTOR; n++ ) {
	if( bVector_[n] ) {

	  if( space == REALSPACE ) {
	    for( int j=0; j<3; j++ ) {  // R, Z, Phi parts.
	      buf = vNode + vFieldStr[n][j];

	      remark( spaceStr[0] + "." + buf );

	      vector_data[n][0][j] =
		get_slice_data( name, spaceStr[0].c_str(), buf.c_str(), dims );
	    }

	    if( nRadial != dims[0] || nTheta != dims[1] || nPhi != dims[2] ) {
	      error( "Error Magnetic Field dimensions do not match Grid dimensions" );
	      error_ = true;
	      return;
	    }
	  } else if( space_ == PERTURBED ) {
	    for( int i=1; i<3; i++ ) { // Real and imaginary perturbed parts
	      for( int j=0; j<3; j++ ) {  // R, Z, Phi parts.
		buf = vNode + vFieldStr[n][j];

		remark( spaceStr[i] + "." + buf );

		vector_data[n][i][j] =
		  get_slice_data( name, spaceStr[i].c_str(), buf.c_str(), dims );

		if( nRadial != dims[0] || nTheta != dims[1] || nMode != dims[2] ) {
		  error( "Error dimensions do not match Grid dimensions" );
		  error_ = true;
		  return;
		}
	      }
	    }
	  }
	}
      }

      if( readAll || readData || modeChange ) {
	StructHexVolMesh *hvm = scinew StructHexVolMesh(nRadial, nTheta, nPhi);

	StructHexVolField<double> *sField[MAX_SCALAR];
	StructHexVolField<Vector> *vField[MAX_VECTOR];

	for( int i=0; i<MAX_SCALAR; i++ ) {
	  if( bScalar_[i] ) {
	    sField[i] = scinew StructHexVolField<double>(hvm, Field::NODE);
	    sHandle_[i] = sField[i];
	  } else {
	    sField[i] = NULL;
	  }
	}

	for( int i=0; i<MAX_VECTOR; i++ ) {
	  if( bVector_[i] ) {
	    vField[i] = scinew StructHexVolField<Vector>(hvm, Field::NODE);
	    vHandle_[i] = vField[i]; 
	  } else {
	    vField[i] = NULL;
	  }
	}

	unsigned int idim = nRadial;
	unsigned int jdim = nTheta;
	unsigned int kdim = nPhi;
	unsigned int mdim = nMode;

	remark( "Creating mesh and field. ");

	// Convert the data and place in the mesh and field.
	double xVal, yVal, zVal, pVal, rad, phi, angle;

	StructHexVolMesh::Node::index_type node;

	register unsigned int i, j, k, l, m, n, index, cc = 0;

	//  Combime the real and imaginary parts.
	if( space_ == PERTURBED ) {
 
	  for( n=0; n<MAX_SCALAR; n++ ) {
	    if( bScalar_[n] )
	      scalar_data[n][0] = scinew double[idim*jdim*kdim];
	  }

	  for( n=0; n<MAX_VECTOR; n++ ) {
	    if( bVector_[n] )
	      for( l=0; l<3; l++ )
		vector_data[n][0][l] = scinew double[idim*jdim*kdim];
	  }

	  if( mode_ == 3 ) {
	    ostringstream str;
	    str << " K Values ";
	    str << grid_data[3][0] << "  ";
	    str << grid_data[3][1] << "  ";
	    str << grid_data[3][2];
	    remark( str.str() );
	  }
	  else {
	    ostringstream str;
	    str << " Mode "  << mode_ << " and K Value " << grid_data[3][mode_];
	    remark( str.str() );
	  }

	  for( k=0; k<kdim; k++ ) {  // Phi loop.

	    phi = grid_data[2][k];

	    for( j=0; j<jdim; j++ ) {  // Theta loop.
	      for( i=0; i<idim; i++ ) {  // R loop.

		for( n=0; n<MAX_SCALAR; n++ ) {
		  if( bScalar_[n] ) {
		    scalar_data[n][0][cc] = 0;

		    //  If summing start at 0 otherwise start with the mode
		    if( mode_ == 3 ) m = 0;
		    else             m = mode_;

		    for( ; m<mdim; m++ ) {  // Mode loop.

		      index = m*idim*jdim + j*idim + i;

		      angle = grid_data[3][m] * phi;

		      scalar_data[n][0][cc] +=
			2.0 * ( cos( angle ) * scalar_data[n][1][index] -
				sin( angle ) * scalar_data[n][2][index] );

		      //  Not summing so quit.
		      if( mode_ < 3 )
			break;
		    }
		  }
		}	   
 
		for( n=0; n<MAX_VECTOR; n++ ) {
		  if( bVector_[n] ) {
		    for( l=0; l<3; l++ )
		      vector_data[n][0][l][cc] = 0;

		    if( mode_ == 3 ) m = 0;
		    else	       m = mode_;

		    for( ; m<mdim; m++ ) {

		      index = m*idim*jdim + j*idim + i;

		      angle = grid_data[3][m] * phi;

		      for( l=0; l<3; l++ )
			vector_data[n][0][l][cc] +=
			  2.0 * ( cos( angle ) * vector_data[n][1][l][index] -
				  sin( angle ) * vector_data[n][2][l][index] );

		      if( mode_ < 3 )
			break;
		    }
		  }
		}
	     
		cc++;
	      }
	    }
	  }
	}

	cc = 0;

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

	      for( n=0; n<MAX_SCALAR; n++ ) {
		if( bScalar_[n] ) {
		  pVal = scalar_data[n][0][cc];
	      
		  sField[n]->set_value(pVal, node);
		}
	      }
	     
	      for( n=0; n<MAX_VECTOR; n++ ) {
		if( bVector_[n] ) {
		  xVal =  vector_data[n][0][0][cc] * cos(phi) -
		    vector_data[n][0][2][cc] * sin(phi);
		  yVal = -vector_data[n][0][0][cc] * sin(phi) -
		    vector_data[n][0][2][cc] * cos(phi);
	      
		  zVal =  vector_data[n][0][1][cc];

		  vField[n]->set_value(Vector(xVal, yVal, zVal), node);
		}
	      }

	      cc++;
	    }
	  }
	}
      }
    
      // Send the intermediate data.
      if( sliceRange_ && slice != sliceStop ) {
	// Get a handle to the output pressure field port.
	for( int n=0; n<MAX_SCALAR; n++ ) {
	  if( sHandle_[n].get_rep() ) {
	    FieldOPort *ofield_port = (FieldOPort *) get_oport(sPortStr[n]);
      
	    if (!ofield_port) {
	      ostringstream str;
	      str << "Unable to initialize " << name << "'s oport";
	      error( str.str() );
	      return;
	    }

	    // Send the data downstream
	    ofield_port->send_intermediate( sHandle_[n] );
	  }
	}

	// Get a handle to the output B Field field port.
	for( int n=0; n<MAX_VECTOR; n++ ) {
	  if( vHandle_[n].get_rep() ) {
	    FieldOPort *ofield_port =
	      (FieldOPort *)get_oport(vPortStr[n]);

	    if (!ofield_port) {
	      ostringstream str;
	      str << "Unable to initialize " << name << "'s oport";
	      error( str.str() );
	      return;
	    }

	    // Send the data downstream
	    ofield_port->send_intermediate( vHandle_[n] );
	  }
	}
      }
    }
  }

  // Get a handle to the output pressure field port.
  for( int n=0; n<MAX_SCALAR; n++ ) {
    if( sHandle_[n].get_rep() ) {
      FieldOPort *ofield_port = (FieldOPort *) get_oport(sPortStr[n]);
      
      if (!ofield_port) {
	ostringstream str;
	str << "Unable to initialize " << name << "'s oport";
	error( str.str() );
	return;
      }

      // Send the data downstream
      ofield_port->send( sHandle_[n] );
    }
  }

  // Get a handle to the output B Field field port.
  for( int n=0; n<MAX_VECTOR; n++ ) {
    if( vHandle_[n].get_rep() ) {
      FieldOPort *ofield_port =
	(FieldOPort *)get_oport(vPortStr[n]);

      if (!ofield_port) {
	ostringstream str;
	str << "Unable to initialize " << name << "'s oport";
	error( str.str() );
	return;
      }

      // Send the data downstream
      ofield_port->send( vHandle_[n] );
    }
  }

#else

  error( "No MDS PLUS availible." );

#endif
}


} // End namespace Fusion
