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
#include <Core/Datatypes/StructHexVolField.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Packages/Fusion/share/share.h>

#include <Packages/Fusion/Core/ThirdParty/mdsPlusReader.h>

#include <algorithm>

namespace Fusion {

#define MAX_GRID 4
#define MAX_SCALAR 1
#define MAX_VECTOR 3

using namespace SCIRun;


static bool pair_less( const pair< int, double> &p1,
		       const pair< int, double> &p2 ) {
  return p1.second < p2.second;
}

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
  GuiString sSliceSkip_;

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
  int sliceSkip_;

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
    sSliceSkip_(context->subVar("sliceSkip")),
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

  for( int i=0; i<MAX_GRID; i++ )
    grid_data[i] = NULL;

  for( int i=0; i<MAX_SCALAR; i++ )
    for( int j=0; j<3; j++ )
      scalar_data[i][j] = NULL;

  for( int i=0; i<MAX_VECTOR; i++ )
    for( int j=0; j<3; j++ )
      for( int k=0; k<3; k++ )
	vector_data[i][j][k] = NULL;
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
  int sliceSkip =  atoi( sSliceSkip_.get().c_str() );
  int space = iSpace_.get();
  int mode = iMode_.get();

  bool readGrid = false, readData = false;
  bool modeChange = false;

  if( bScalar_[0] != iScalar0_.get() ) {
    bScalar_[0] = iScalar0_.get();

    if( bScalar_[0] ) readData = true;
  }

  if( bVector_[0] != iVector0_.get() ) {
    bVector_[0] = iVector0_.get();

    if( bVector_[0] ) readData = true;
  }

  if( bVector_[1] != iVector1_.get() ) {
    bVector_[1] = iVector1_.get();

    if( bVector_[1] ) readData = true;
  }

  if( bVector_[2] != iVector2_.get() ) {
    bVector_[2] = iVector2_.get();

    if( bVector_[2] ) readData = true;
  }

  bool readSomething = false;

  for( int i=0; i<MAX_SCALAR; i++ ) {
    if( bScalar_[i] ) {
      readSomething = true;
      break;
    }
  }

  for( int i=0; i<MAX_VECTOR; i++ ) {
    if( bVector_[i] ) {
      readSomething = true;
      break;
    }
  }

  if( ! readSomething ) {
    error( "no data selected to be read." );
    error_ = true;
    return;
  }

  if( error_  == true ||
      
      server_ != server ||
      tree_   != tree   ||
      shot_   != shot ) {

    error_ = false;

    server_ = server;
    tree_   = tree;
    shot_   = shot;

    readGrid = readData = true;
  }

  if( sliceRange_ != sliceRange )
      sliceRange_ = sliceRange;

  if( sliceRange_ == true ||
      slice_ != slice ||
      space_ != space) {

    slice_  = slice;
    space_ = space;

    sliceStart_ = sliceStart;
    sliceStop_ = sliceStop;
    sliceSkip_ = sliceSkip;

    readData = true;
  }
    
  if( mode_ != mode ) {
    mode_ = mode;
    modeChange = true;
  }
  
  MDSPlusReader mds;

  if( readGrid || readData ) {

    int trys = 0;
    int retVal;

    while( trys < 10 && (retVal = mds.connect(server.c_str()) ) == -2 ) {
      remark( "Waiting for the connection to become free." );
      sleep( 1 );
      trys++;
    }
  
    /* Connect to MDSplus */
    if( retVal == -2 ) {
      error( "Connection to Mds Server " + server + " too busy ... giving up.");
      error_ = true;
      return;
    }
    else if( retVal < 0 ) {
      error( "Connecting to Mds Server " + server );
      error_ = true;
      return;
    }
    else
      remark( "Conecting to MdsPlus Server " + server );

    // Open tree
    trys = 0;

    while( trys < 10 && (retVal = mds.open( tree.c_str(), shot) ) == -2 ) {
      remark( "Waiting for the tree and shot to become free." );
      sleep( 1 );
      trys++;
    }

    if( retVal == -2 ) {
      ostringstream str;
      str << "Can not open " << tree << " tree and shot " << shot << " too busy ... giving up.";
      error( str.str() );
      error_ = true;
      return;
    }

    else if( retVal < 0 ) {
      ostringstream str;
      str << "Opening " << tree << " tree and shot " << shot;
      error( str.str() );
      error_ = true;
      return;
    }
    else {
      ostringstream str;
      str << "Opening " << tree << " tree and shot " << shot;
      remark( str.str() );
    }
  }

  // If reading the same tree no need to get the grid.
  if( readGrid ) {

    int dims[3];

    for( int n=0; n<MAX_GRID; n++ ) {
      if( grid_data[n] ) delete grid_data[n];
      grid_data[n] = NULL;
    }

    // Query the server for the cylindrical components of the grid.
    for( int n=0; n<MAX_GRID; n++ ) {

      dims[0] = dims[1] = dims[2] = 0;

      remark( gridStr[n] );
      grid_data[n] = mds.grid( gridStr[n].c_str(), dims );

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
	    mds.disconnect();
	    return;
	  } else {
	    break;
	  }

	case 2:
	  nPhi = dims[0];
	  break;
	
	case 3:
	  nMode = dims[0];
	  for( int i=0; i<3; i++ )
	    if( grid_data[n][i] != (double) i ) {

	      ostringstream str;
	      str << "Invalid mode value " << grid_data[n][i];
	      str << " ... correcting to " << i;
	      remark( str.str() );
	      grid_data[n][i] = (double) i;
	    }
	  break;
	}
      } else {
	error( "Can not get Grid data" );
	error_ = true;
	mds.disconnect();
	return;
      }
    }

    {
      ostringstream str;
      str << "Grid size: ";
      str << nRadial << " " << nTheta << " " << nPhi << " " << nMode;
      remark( str.str() );
    }
  }


  if( readData || modeChange ) {

    std::string sNode("SCALARS");                 // Node/Data to fetch
    std::string vNode("VECTORS.CYLINDRICAL");     // Node/Data to fetch

    std::string buf;

    int dims[3];

    std::string name;           // Used to hold the name of the slice 
    double time;                // Used to hold the time of the slice 
    int nSlices;                // Number of time slices.
    int *nids;                  // IDs to the time slice nodes in the tree 

    // Query the server for the number of slices in the tree
    nSlices = mds.slice_ids( &nids );

    if( readData ) {
      ostringstream str;
      str << "Number of slices in the current shot " << nSlices;
      remark( str.str() );
    }

    vector< pair< int, double > > sliceList;

    for( int ic=0; ic<nSlices; ic++ ) {
      name = mds.slice_name( nids, ic );
      time = mds.slice_time( name );

      sliceList.push_back( pair< int, double >(ic, time) );
    }

    std::sort( sliceList.begin(), sliceList.end(), pair_less );

    if( sliceRange_ ) {

      if( sliceStart_< 0 || nSlices<=sliceStop_ ) {
	ostringstream str;
	str << "Slice range is outside of the range [0-" << nSlices << ").";
	error( str.str() );
	error_ = true;
	mds.disconnect();
	return;
      } else if( sliceStop_ < sliceStart_ ) {
	ostringstream str;
	str << "Improper slice range";
	error( str.str() );
	error_ = true;
	mds.disconnect();
	return;
      } else if( sliceSkip_< 0 || sliceStop_-sliceStart_<sliceSkip_ ) {
	ostringstream str;
	str << "Slice skip is outside of the range [0-" << sliceStop_-sliceStart_ << ").";
	error( str.str() );
	error_ = true;
	mds.disconnect();
	return;
      }
    } else {

      if( slice_< 0 || nSlices<=slice_ ) {
	ostringstream str;
	str << "Slice " << slice_ << " is outside of the range [0-" << nSlices << ").";
	error( str.str() );
	error_ = true;
	mds.disconnect();
	return;
      }

      sliceStart = slice;
      sliceStop  = slice;
    }

    for( int ic=sliceStart; ic<=sliceStop; ic+=sliceSkip ) {

      slice = sliceList[ic].first;

      name = mds.slice_name( nids, slice );
      time = mds.slice_time( name );

      {
	ostringstream str;
	str << "Processing slice " << name << " at time " << time;
	remark( str.str() );
      }


      if( sliceRange_ ) {
	slice_ = ic;

	ostringstream str;
	str << slice_;

	sSliceNumber_.set( str.str().c_str() );
      }

      if( readData  ) {
	// Fetch the Scalar data from the node
	for( int n=0; n<MAX_SCALAR; n++ ) {

	  for( int i=0; i<3; i++ ) {
	    if( scalar_data[n][i] ) delete scalar_data[n][i];
	    scalar_data[n][i] = NULL;
	  }

	  if( bScalar_[n] ) {
	
	    buf = sNode + sFieldStr[n];

	    int i=0, ncomps=0;
	    
	    if( space == REALSPACE ) {
	      i = 0;
	      ncomps = i+1;
	    } else if( space_ == PERTURBED ) {
	      i = 1;
	      ncomps = i+2;
	    }

	    for( ; i<ncomps; i++ ) { // Real and imaginary perturbed parts
	      remark( spaceStr[i] + "." + buf );

	      scalar_data[n][i] =
		mds.slice_data( name, spaceStr[i].c_str(), buf.c_str(), dims );

	      if( scalar_data[n][i] == NULL ) {
		error( "Error can not get Scalar data" );
		if( ! sliceRange_ ) {
		  error_ = true;
		  mds.disconnect();
		  return;
		}
	      } else if( nRadial != dims[0] ||
			 nTheta  != dims[1] ||
			 nPhi    != dims[2] ) {
	      
		ostringstream str;
		str << "Error dimensions do not match Grid dimensions: ";
		str << dims[0] << " " << dims[1] << " " << dims[2];
		error( str.str() );
		if( ! sliceRange_ ) {
		  error_ = true;
		  mds.disconnect();
		  return;
		}
	      }
	    }
	  }
	}

	// Fetch the Vector data from the node
	for( int n=0; n<MAX_VECTOR; n++ ) {

	  for( int i=0; i<3; i++ ) {
	    for( int j=0; j<3; j++ ) {
	      if( vector_data[n][i][j] ) delete vector_data[n][i][j];
	      vector_data[n][i][j] = NULL;
	    }
	  }

	  if( bVector_[n] ) {

	    int i=0, ncomps=0;
	    
	    if( space == REALSPACE ) {
	      i = 0;
	      ncomps = i+1;
	    } else if( space_ == PERTURBED ) {
	      i = 1;
	      ncomps = i+2;
	    }

	    for( ; i<ncomps; i++ ) { // Real and imaginary perturbed parts
	      for( int j=0; j<3; j++ ) {  // R, Z, Phi parts.
		buf = vNode + vFieldStr[n][j];

		remark( spaceStr[i] + "." + buf );

		vector_data[n][i][j] =
		  mds.slice_data( name, spaceStr[i].c_str(), buf.c_str(), dims );

		if( vector_data[n][i][j] == NULL ) {
		  error( "Error can not get Vector data" );
		  if( ! sliceRange_ ) {
		    error_ = true;
		    mds.disconnect();
		    return;
		  }
		} else if( nRadial != dims[0] ||
			   nTheta  != dims[1] ||
			   nPhi    != dims[2] ) {

		  ostringstream str;
		  str << "Error dimensions do not match Grid dimensions: ";
		  str << dims[0] << " " << dims[1] << " " << dims[2];
		  error( str.str() );
		  if( ! sliceRange_ ) {
		    error_ = true;
		    mds.disconnect();
		    return;
		  }
		}
	      }
	    }
	  }
	}
      }

      if( readData || modeChange ) {
	StructHexVolMesh *hvm = scinew StructHexVolMesh(nRadial, nTheta-1, nPhi-1);

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
	    str << " Summing K Values ";
	    str << grid_data[3][0] << "  ";
	    str << grid_data[3][1] << "  ";
	    str << grid_data[3][2];
	    remark( str.str() );
	  }
	  else {
	    ostringstream str;
	    str << "Calculating Mode "  << mode_ << " and K Value " << grid_data[3][mode_];
	    remark( str.str() );
	  }

	  for( k=0; k<kdim-1; k++ ) {  // Phi loop.

	    phi = grid_data[2][k];

	    for( j=0; j<jdim-1; j++ ) {  // Theta loop.
	      for( i=0; i<idim; i++ ) {  // R loop.

		for( n=0; n<MAX_SCALAR; n++ ) {
		  if( bScalar_[n] ) {
		    scalar_data[n][0][cc] = 0;

		    //  If summing start at 0 otherwise start with the mode
		    if( mode_ == 3 ) m = 0;
		    else             m = mode_;

		    for( ; m<mdim; m++ ) {  // Mode loop.

		      index = m*idim*jdim + j*idim + i;

		      angle = grid_data[3][m] * phi; // Mode * phi slice.

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

		    //  If summing start at 0 otherwise start with the mode
		    if( mode_ == 3 ) m = 0;
		    else	     m = mode_;

		    for( ; m<mdim; m++ ) {  // Mode loop.

		      index = m*idim*jdim + j*idim + i;

		      angle = grid_data[3][m] * phi; // Mode * phi slice.

		      for( l=0; l<3; l++ )
			vector_data[n][0][l][cc] +=
			  2.0 * ( cos( angle ) * vector_data[n][1][l][index] -
				  sin( angle ) * vector_data[n][2][l][index] );

		      //  Not summing so quit.
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

	remark( "Creating mesh and field. ");

	cc = 0;

	int jIndex;

	for( k=0; k<kdim-1; k++ ) {

	  node.k_ = k;

	  phi = grid_data[2][k];

	  for( j=0; j<jdim-1; j++ ) {

	    node.j_ = j;

	    jIndex = j * idim;

	    for( i=0; i<idim; i++ ) {

	      index = i + jIndex;

	      node.i_ = i;

	      rad = grid_data[0][index];

	      xVal =  rad * cos( phi );
	      yVal = -rad * sin( phi );
	      zVal =  grid_data[1][index];

	      hvm->set_point( Point( xVal, yVal, zVal ), node );

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

		  if( xVal * xVal + yVal * yVal + zVal * zVal < 1.0e-24 ) {
		    remark( "Replaced a zero length vector" );
		    xVal = yVal = zVal = 1.0e-12;
		  }

		  vField[n]->set_value(Vector(xVal, yVal, zVal), node);
		}
	      }

	      cc++;
	    }
	  }
	}
      }
    
      // Send the intermediate data.
      if( sliceRange_ && ic != sliceStop ) {
	// Get a handle to the output pressure field port.
	for( int n=0; n<MAX_SCALAR; n++ ) {
	  if( sHandle_[n].get_rep() ) {
	    FieldOPort *ofield_port = (FieldOPort *) get_oport(sPortStr[n]);
      
	    if (!ofield_port) {
	      ostringstream str;
	      str << "Unable to initialize " << name << "'s oport";
	      error( str.str() );
	      if( readGrid || readData )
		mds.disconnect();
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
	      if( readGrid || readData )
		mds.disconnect();
	      return;
	    }

	    // Send the data downstream
	    ofield_port->send_intermediate( vHandle_[n] );
	  }
	}
      }
    }
  }

  if( readGrid || readData )
    mds.disconnect();

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
