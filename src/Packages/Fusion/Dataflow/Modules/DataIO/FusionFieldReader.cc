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
 *  FusionFieldReader.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>

#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>

#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/HexVolField.h>
#include <Packages/Fusion/share/share.h>

#include <fstream>
#include <sys/stat.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE FusionFieldReader : public Module {
public:
  FusionFieldReader(const string& id);

  virtual ~FusionFieldReader();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);

private:
  GuiString filename_;
  string old_filename_;
  time_t old_filemodification_;

  FieldHandle  fHandle_;
  MatrixHandle mbHandle_;
  MatrixHandle mvHandle_;
  MatrixHandle mpHandle_;
};

extern "C" FusionSHARE Module* make_FusionFieldReader(const string& id) {
  return scinew FusionFieldReader(id);
}

FusionFieldReader::FusionFieldReader(const string& id)
  : Module("FusionFieldReader", id, Source, "DataIO", "Fusion"),
    filename_("filename", id, this)
{
}

FusionFieldReader::~FusionFieldReader(){
}

void FusionFieldReader::execute(){

  //
  enum DATA_TYPE { UNKNOWN=0, NIMROD=1 };

  //                 CARTESIAN     CYLINDRICAL       Fusion
  enum DATA_FIELDS { GRID_XYZ=0,   GRID_R_PHI_Z=1,   GRID_R_Z_PHI=2,
		     BFIELD_XYZ=3, BFIELD_R_PHI_Z=4, BFIELD_R_Z_PHI=5,
		     VFIELD_XYZ=6, VFIELD_R_PHI_Z=7, VFIELD_R_Z_PHI=8,
		     PRESSURE = 9,
		     MAX_FIELDS = 10};

  int *readOrder = new int[ (int) MAX_FIELDS ];

  for( int i=0; i<(int)MAX_FIELDS; i++ )
    readOrder[i] = -1;

      
  string new_filename(filename_.get());
  
  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
  if (stat(new_filename.c_str(), &buf)) {
    error( string("File not found ") + new_filename );
    return;
  }

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif

  if( !fHandle_.get_rep() ||
      new_filename         != old_filename_ || 
      new_filemodification != old_filemodification_)
  {
    cout << "Reading the file " <<  new_filename << endl;

    old_filemodification_ = new_filemodification;
    old_filename_         = new_filename;

    // Open the mesh file.
    ifstream ifs( new_filename.c_str() );

    if (!ifs) {
      error( string("Could not open file ") + new_filename );
      return;
    }
	
    char token[32];

    DATA_TYPE dataType = UNKNOWN;

    // Read the slice type.
    ifs >> token;

    if( strcmp( token, "UNKNOWN" ) ) {
      dataType = UNKNOWN;
    }
    else if( strcmp( token, "NIMROD" ) ) {
      dataType = NIMROD;
    }
    else
    {
      warning( string("Unknown data type: expected NIMROD but read " ) + token );
    }

    // Read the slice time.
    ifs >> token;

    if( strcmp( token, "TIME" ) ) {
      error( string("Bad token: expected TIME but read ") + token );
      return;
    }

    double time;

    ifs >> time;

    // Read the field data.
    ifs >> token;

    if( strcmp( token, "FIELDS" ) ) {
      error( string("Bad token: expected FILEDS but read ") + token );
      return;
    }


    // Get the number values being read.
    int nVals;

    ifs >> nVals;

    cout << "FusionFieldReader - Number of values including grid " << nVals << endl;

    int index = 0;

    while( index < nVals )
    {
      ifs >> token;

      cout << "FusionFieldReader - Token and index " << token << " " << index << endl;

      if( strcmp( token, "GRID_X_Y_Z" ) == 0 ) {
	readOrder[GRID_XYZ] = index;
	index += 3;
      }

      else if( strcmp( token, "GRID_R_PHI_Z" ) == 0 ) {
	readOrder[GRID_R_PHI_Z] = index;
	index += 3;
      }

      else if( strcmp( token, "GRID_R_Z_PHI" ) == 0 ) {
	readOrder[GRID_R_Z_PHI] = index;
	index += 3;
      }


      // B Field
      else if( strcmp( token, "BFIELD_X_Y_Z" ) == 0 ) {
	readOrder[BFIELD_XYZ] = index;
	index += 3;
      }

      else if( strcmp( token, "BFIELD_R_PHI_Z" ) == 0 ) {
	readOrder[BFIELD_R_PHI_Z] = index;
	index += 3;
      }

      else if( strcmp( token, "BFIELD_R_Z_PHI" ) == 0 ) {
	readOrder[BFIELD_R_Z_PHI] = index;
	index += 3;
      }


      // Velocity
      else if( strcmp( token, "VFIELD_X_Y_Z" ) == 0 ) {
	readOrder[VFIELD_XYZ] = index;
	index += 3;
      }

      else if( strcmp( token, "VFIELD_R_PHI_Z" ) == 0 ) {
	readOrder[VFIELD_R_PHI_Z] = index;
	index += 3;
      }

      else if( strcmp( token, "VFIELD_R_Z_PHI" ) == 0 ) {
	readOrder[VFIELD_R_Z_PHI] = index;
	index += 3;
      }

      // Pressure
      else if( strcmp( token, "PRESSURE" ) == 0 ) {
	readOrder[PRESSURE] = index;
	index += 1;
      }

      else
      { 
	error( string("Bad token: expected XXX but read ") + token );

	return;
      }
    }

    // Get the dimensions of the grid.
    // Read the field data.
    ifs >> token;

    if( strcmp( token, "DIMENSIONS" ) ) {
      error( string("Bad token: expected DIMENSIONS but read ") + token );
      return;
    }

    int idim, jdim, kdim;
    ifs >> idim >> jdim >> kdim; 

    // The Fusion data repeats in the theta and phi directions but
    // it is accounted for in the Mesh generation.
    //jdim--;
    //kdim--;

    cout << "FusionFieldReader - " << idim << "  " << jdim << "  " << kdim << endl;

    // Create the grid, and scalar and vector data matrices.
    HexVolMesh *hvm = scinew HexVolMesh;

    DenseMatrix  *bMatrix = NULL;
    DenseMatrix  *vMatrix = NULL;
    ColumnMatrix *pMatrix = NULL;

    if( readOrder[BFIELD_XYZ    ] > -1 ||
	readOrder[BFIELD_R_PHI_Z] > -1 ||
	readOrder[BFIELD_R_Z_PHI] > -1 )
      bMatrix = scinew DenseMatrix(idim*jdim*kdim,3);

    if( readOrder[VFIELD_XYZ    ] > -1 ||
	readOrder[VFIELD_R_PHI_Z] > -1 ||
	readOrder[VFIELD_R_Z_PHI] > -1 )
      vMatrix = scinew DenseMatrix(idim*jdim*kdim,3);

    if( readOrder[PRESSURE] > -1 )
      pMatrix = scinew ColumnMatrix(idim*jdim*kdim);


    // Read the data.
    double* data = new double[nVals];

    double xVal, yVal, zVal, pVal, rad, phi;

    int cc = 0;

    bool phi_read;

    for( int k=0; k<kdim; k++ ) {
      for( int j=0; j<jdim; j++ ) {
	for( int i=0; i<idim; i++ ) {

	  // Read the data in no matter the order.
	  for( index=0; index<nVals; index++ ) {
	    ifs >> data[index];
			

	    if( ifs.eof() ) {
	      error( string("Could not read grid ") + new_filename );

	      cerr << "FusionFieldReader " << i << "  " << j << "  " << k << "  " << index << endl;

	      return;
	    }
	  }

	  phi_read = false;

	  // Grid
	  if( readOrder[GRID_XYZ] > -1 ) {

	    index = readOrder[GRID_XYZ];

	    xVal = data[index  ];
	    yVal = data[index+1];
	    zVal = data[index+2];

	    hvm->add_point( Point( xVal, yVal, zVal ) );
	  }

	  else if( readOrder[GRID_R_PHI_Z] > -1 ) {

	    index = readOrder[GRID_R_PHI_Z];

	    rad = data[index  ];
	    phi = data[index+1];

	    phi_read = true;

	    xVal =  rad * cos( phi );
	    yVal = -rad * sin( phi );
	    zVal =  data[index+2];

	    hvm->add_point( Point( xVal, yVal, zVal ) );
	  }

	  else if( readOrder[GRID_R_Z_PHI] > -1 ) {

	    index = readOrder[GRID_R_Z_PHI];

	    rad = data[index  ];
	    phi = data[index+2];

	    phi_read = true;

	    xVal =  rad * cos( phi );
	    yVal = -rad * sin( phi );
	    zVal =  data[index+1];

	    hvm->add_point( Point( xVal, yVal, zVal ) );
	  }


	  // B Field
	  if( readOrder[BFIELD_XYZ] > -1 ) {

	    index = readOrder[BFIELD_XYZ];

	    xVal = data[index  ];
	    yVal = data[index+1];
	    zVal = data[index+2];

	    bMatrix->put(cc, 0, xVal);
	    bMatrix->put(cc, 1, yVal);
	    bMatrix->put(cc, 2, zVal);
	  }
	  else if( readOrder[BFIELD_R_PHI_Z] > -1 ) {

	    index = readOrder[BFIELD_R_PHI_Z];

	    if( phi_read ) {

	      xVal =  data[index  ] * cos(phi) -
		data[index+1] * sin(phi);
	      yVal = -data[index  ] * sin(phi) -
		data[index+1] * cos(phi );
	      zVal =  data[index+2];

	      bMatrix->put(cc, 0, xVal);
	      bMatrix->put(cc, 1, yVal);
	      bMatrix->put(cc, 2, zVal);
	    }
	  }

	  else if( readOrder[BFIELD_R_Z_PHI] > -1 ) {

	    index = readOrder[BFIELD_R_Z_PHI];

	    if( phi_read ) {

	      xVal =  data[index  ] * cos(phi) -
		data[index+2] * sin(phi);
	      yVal = -data[index  ] * sin(phi) -
		data[index+2] * cos(phi );
	      zVal =  data[index+1];

	      bMatrix->put(cc, 0, xVal);
	      bMatrix->put(cc, 1, yVal);
	      bMatrix->put(cc, 2, zVal);
	    }
	  }

	  // V Field
	  if( readOrder[VFIELD_XYZ] > -1 ) {

	    index = readOrder[VFIELD_XYZ];

	    xVal = data[index  ];
	    yVal = data[index+1];
	    zVal = data[index+2];

	    vMatrix->put(cc, 0, xVal);
	    vMatrix->put(cc, 1, yVal);
	    vMatrix->put(cc, 2, zVal);
	  }
	  else if( readOrder[VFIELD_R_PHI_Z] > -1 ) {

	    index = readOrder[VFIELD_R_PHI_Z];


	    if( phi_read ) {

	      xVal =  data[index  ] * cos(phi) -
		data[index+1] * sin(phi);
	      yVal = -data[index  ] * sin(phi) -
		data[index+1] * cos(phi);
	      zVal =  data[index+2];

	      vMatrix->put(cc, 0, xVal);
	      vMatrix->put(cc, 1, yVal);
	      vMatrix->put(cc, 2, zVal);
	    }
	  }

	  else if( readOrder[VFIELD_R_Z_PHI] > -1 ) {

	    index = readOrder[VFIELD_R_Z_PHI];


	    if( phi_read ) {

	      xVal =  data[index  ] * cos(phi) -
		data[index+2] * sin(phi);
	      yVal = -data[index  ] * sin(phi) -
		data[index+2] * cos(phi);
	      zVal =  data[index+1];

	      vMatrix->put(cc, 0, xVal);
	      vMatrix->put(cc, 1, yVal);
	      vMatrix->put(cc, 2, zVal);
	    }
	  }

	  // Pressure
	  if( readOrder[PRESSURE] > -1 ) {

	    index = readOrder[PRESSURE];

	    pVal = data[index];

	    pMatrix->put(cc, 0, pVal);
	  }

	  cc++;
	}
      }
    }

    // Make sure that all of the data was read.
    string tmpStr;

    ifs >> tmpStr;

    if( !ifs.eof() ) {
      error( string("not all data was read ") + new_filename );
      error( string("not all data was read ") + tmpStr );
    }

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

    // Now after the mesh has been created, create the field.
    HexVolField<double> *hvf =
      scinew HexVolField<double>(HexVolMeshHandle(hvm), Field::NODE);

    hvm->set_property( "I Dim", idim, false );
    hvm->set_property( "J Dim", jdim, false );
    hvm->set_property( "K Dim", kdim, false );


    fHandle_  = FieldHandle( hvf );

    if( bMatrix ) mbHandle_ = MatrixHandle( bMatrix );
    if( vMatrix ) mvHandle_ = MatrixHandle( vMatrix );
    if( pMatrix ) mpHandle_ = MatrixHandle( pMatrix );
  }
  else
  {
    cout << "Already read the file " <<  new_filename << endl;
  }


  // Get a handle to the output field port.
  {
    FieldOPort *ofield_port = 
      (FieldOPort *) get_oport("Output Grid Field");

    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( fHandle_ );
  }


  // Get a handle to the output pressure matrix port.
  if( mpHandle_.get_rep() )
  {
    MatrixOPort *omatrix_port =
      (MatrixOPort *) get_oport("Output Pressure Scalar");

    if (!omatrix_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    omatrix_port->send( mpHandle_ );
  }

  // Get a handle to the output B Field matrix port.
  if( mbHandle_.get_rep() )
  {
    MatrixOPort *omatrix_port =
      (MatrixOPort *)get_oport("Output B Field Vector");

    if (!omatrix_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    omatrix_port->send( mbHandle_ );
  }

  // Get a handle to the output V Field matrix port.
  if( mvHandle_.get_rep() )
  {
    MatrixOPort *omatrix_port =
      (MatrixOPort *) get_oport("Output V Field Vector");

    if (!omatrix_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    omatrix_port->send( mvHandle_ );
  }
}


void FusionFieldReader::tcl_command(TCLArgs& args, void* userdata) {
  Module::tcl_command(args, userdata);
}

} // End namespace Fusion
