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
#include <Core/Datatypes/StructHexVolField.h>

#include <Core/Math/Trig.h>

#include <Packages/Fusion/share/share.h>

#include <fstream>
#include <sys/stat.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE FusionFieldReader : public Module {
public:
  FusionFieldReader(GuiContext *context);

  virtual ~FusionFieldReader();

  virtual void execute();

private:
  GuiString filename_;
  string old_filename_;
  time_t old_filemodification_;

  FieldHandle  bHandle_;
  FieldHandle  vHandle_;
  FieldHandle  pHandle_;
  FieldHandle  tHandle_;

  FieldHandle  eHandle_;
};


DECLARE_MAKER(FusionFieldReader)


FusionFieldReader::FusionFieldReader(GuiContext *context)
  : Module("FusionFieldReader", context, Source, "DataIO", "Fusion"),
    filename_(context->subVar("filename"))
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
		     PRESSURE = 9, TEMPERATURE= 10,
		     MAX_FIELDS = 11};

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

  if( new_filename         != old_filename_ || 
      new_filemodification != old_filemodification_)
  {
    remark( "Reading the file " +  new_filename );

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

    remark( "Number of values including grid " + nVals );

    int index = 0;

    while( index < nVals )
    {
      ifs >> token;

      //      remark( "Token and index " + token + " " + index );

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

      // Temperature
      else if( strcmp( token, "TEMPERATURE" ) == 0 ) {
	readOrder[TEMPERATURE] = index;
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

    // Read whether the data is wrapped our clamped
    bool repeated = false;

    ifs >> token;

    if( strcmp( token, "REPEATED" ) == 0 )
      repeated = true;
    else if( strcmp( token, "SINGULAR" ) == 0 )
      repeated = false;
    else {
      error( string("Bad token: expected REPEATED or SINGULAR but read ") + token );
      return;
    }

    unsigned int idim, jdim, kdim;

    ifs >> idim >> jdim >> kdim;
    
    

    //    remark( idim + "  " + jdim + "  " + kdim );
 
    // Create the grid, and scalar and vector data matrices.
    StructHexVolMesh *hvm = NULL;

    if( readOrder[GRID_XYZ    ] > -1 ||
	readOrder[GRID_R_PHI_Z] > -1 ||
	readOrder[GRID_R_Z_PHI] > -1 ) {
    
      if( repeated )
	hvm = scinew StructHexVolMesh(idim, jdim-1, kdim-1);
      else
	hvm = scinew StructHexVolMesh(idim, jdim, kdim);
    }
    else {
      error( string("No grid present - unable to create the field(s).") );
      return;
    }

    // Now after the mesh has been created, create the field.
    StructHexVolField<Vector> *bfield = NULL;
    StructHexVolField<Vector> *vfield = NULL;
    StructHexVolField<double> *pfield = NULL;
    StructHexVolField<double> *tfield = NULL;
    StructHexVolField< vector<double> > *efield = NULL;

    if( readOrder[BFIELD_XYZ    ] > -1 ||
	readOrder[BFIELD_R_PHI_Z] > -1 ||
	readOrder[BFIELD_R_Z_PHI] > -1 ) {
      bfield = scinew StructHexVolField<Vector>(hvm, Field::NODE);
      bHandle_ = bfield;
    }
    if( readOrder[VFIELD_XYZ    ] > -1 ||
	readOrder[VFIELD_R_PHI_Z] > -1 ||
	readOrder[VFIELD_R_Z_PHI] > -1 ) {
      vfield = scinew StructHexVolField<Vector>(hvm, Field::NODE);
      vHandle_ = vfield;

      efield = scinew StructHexVolField< vector<double> >(hvm, Field::NODE);
      eHandle_ = efield;
    }

    if( readOrder[PRESSURE] > -1 ) {
      pfield = scinew StructHexVolField<double>(hvm, Field::NODE);
      pHandle_ = pfield;
    }

    if( readOrder[TEMPERATURE] > -1 ) {
      tfield = scinew StructHexVolField<double>(hvm, Field::NODE);
      tHandle_ = tfield;
    }

    // Read the data.
    double* data = new double[nVals];

    double xVal, yVal, zVal, pVal, tVal, rad, phi;

    bool phi_read;

    StructHexVolMesh::Node::index_type node;

    register unsigned int i, j, k;

    double b_phi_min =  9999.9;
    double b_phi_max = -9999.9;

    for( k=0; k<kdim - repeated ? 1 : 0; k++ ) {
      for( j=0; j<jdim - repeated ? 1 : 0; j++ ) {
	for( i=0; i<idim; i++ ) {

	  // Read the data in no matter the order.
	  for( index=0; index<nVals; index++ ) {
	    ifs >> data[index];			

	    if( ifs.eof() ) {
	      error( string("Could not read grid ") + new_filename );

//	      error( "FusionFieldReader " + i + "  " + j + "  " + k + "  " + index );

	      return;
	    }
	  }

	  phi_read = false;

	  node.i_ = i;
	  node.j_ = j;
	  node.k_ = k;

	  // Grid
	  if( readOrder[GRID_XYZ] > -1 ) {

	    index = readOrder[GRID_XYZ];

	    xVal = data[index  ];
	    yVal = data[index+1];
	    zVal = data[index+2];

	    hvm->set_point(Point(xVal, yVal, zVal), node);
	  }

	  else if( readOrder[GRID_R_PHI_Z] > -1 ) {

	    index = readOrder[GRID_R_PHI_Z];

	    rad = data[index  ];
	    phi = data[index+1];

	    phi_read = true;

	    xVal =  rad * cos( phi );
	    yVal = -rad * sin( phi );
	    zVal =  data[index+2];

	    hvm->set_point(Point(xVal, yVal, zVal), node);
	  }

	  else if( readOrder[GRID_R_Z_PHI] > -1 ) {

	    index = readOrder[GRID_R_Z_PHI];

	    rad = data[index  ];
	    phi = data[index+2];

	    phi_read = true;

	    xVal =  rad * cos( phi );
	    yVal = -rad * sin( phi );
	    zVal =  data[index+1];

/* Experiment for working in cylindrical coordinates.
	    hvm->set_point( Point( data[index  ],
				   data[index+1],
				   data[index+2] ), node );
*/
	    hvm->set_point(Point(xVal, yVal, zVal), node);
	  }


	  // B Field
	  if( readOrder[BFIELD_XYZ] > -1 ) {

	    index = readOrder[BFIELD_XYZ];

	    xVal = data[index  ];
	    yVal = data[index+1];
	    zVal = data[index+2];

	    if( xVal * xVal + yVal * yVal + zVal * zVal < 1.0e-24 ) {
	      remark( "Replaced a zero length vector" );
	      xVal = yVal = zVal = 1.0e-12;
	    }

	    bfield->set_value(Vector(xVal, yVal, zVal), node);
	  }
	  else if( readOrder[BFIELD_R_PHI_Z] > -1 ) {

	    index = readOrder[BFIELD_R_PHI_Z];

	    if( phi_read ) {

	      xVal =  data[index  ] * cos(phi) - data[index+1] * sin(phi);
	      yVal = -data[index  ] * sin(phi) - data[index+1] * cos(phi);
	      zVal =  data[index+2];

	      if( xVal * xVal + yVal * yVal + zVal * zVal < 1.0e-24 ) {
		remark( "Replaced a zero length vector" );
		xVal = yVal = zVal = 1.0e-12;
	      }

	      bfield->set_value(Vector(xVal, yVal, zVal), node);
	    }
	  }

	  else if( readOrder[BFIELD_R_Z_PHI] > -1 ) {

	    index = readOrder[BFIELD_R_Z_PHI];

	    if( phi_read ) {

	      xVal =  data[index ] * cos(phi) - data[index+2] * sin(phi);
	      yVal = -data[index ] * sin(phi) - data[index+2] * cos(phi);
	      zVal =  data[index+1];

/* Experiment for working in cylindrical coordinates.
	      if( b_phi_min > data[index+2] )
		b_phi_min = data[index+2];

	      else if( b_phi_max < data[index+2] )
		b_phi_max = data[index+2];

	      rad = sqrt( xVal * xVal + yVal * yVal );
	      phi = atan2( -yVal, xVal );

	      if( phi < 0 )
		phi += 2.0 * PI;

	      xVal = rad;
	      yVal = zVal;
	      zVal = phi;
*/
	    if( xVal * xVal + yVal * yVal + zVal * zVal < 1.0e-24 ) {
	      remark( "Replaced a zero length vector" );
	      xVal = yVal = zVal = 1.0e-12;
	    }

	    bfield->set_value(Vector(xVal, yVal, zVal), node);
	    }
	  }

	  // V Field
	  if( readOrder[VFIELD_XYZ] > -1 ) {

	    index = readOrder[VFIELD_XYZ];

	    xVal = data[index  ];
	    yVal = data[index+1];
	    zVal = data[index+2];

	    if( xVal * xVal + yVal * yVal + zVal * zVal < 1.0e-24 ) {
	      remark( "Replaced a zero length vector" );
	      xVal = yVal = zVal = 1.0e-12;
	    }

	    vfield->set_value(Vector(xVal, yVal, zVal), node);

	    vector< double > errors;
	    errors.push_back( xVal );
	    errors.push_back( yVal );
	    errors.push_back( zVal );

	    efield->set_value(errors, node);
	  }
	  else if( readOrder[VFIELD_R_PHI_Z] > -1 ) {

	    index = readOrder[VFIELD_R_PHI_Z];


	    if( phi_read ) {

	      xVal =  data[index  ] * cos(phi) - data[index+1] * sin(phi);
	      yVal = -data[index  ] * sin(phi) - data[index+1] * cos(phi);
	      zVal =  data[index+2];

	      if( xVal * xVal + yVal * yVal + zVal * zVal < 1.0e-24 ) {
		remark( "Replaced a zero length vector" );
		xVal = yVal = zVal = 1.0e-12;
	      }

	      vfield->set_value(Vector(xVal, yVal, zVal), node);

	      vector< double > errors;
	      errors.push_back( xVal );
	      errors.push_back( yVal );
	      errors.push_back( zVal );
	      
	      efield->set_value(errors, node);
	    }
	  }

	  else if( readOrder[VFIELD_R_Z_PHI] > -1 ) {

	    index = readOrder[VFIELD_R_Z_PHI];


	    if( phi_read ) {
	      xVal =  data[index  ] * cos(phi) - data[index+2] * sin(phi);
	      yVal = -data[index  ] * sin(phi) - data[index+2] * cos(phi);
	      zVal =  data[index+1];

	      if( xVal * xVal + yVal * yVal + zVal * zVal < 1.0e-24 ) {
		remark( "Replaced a zero length vector" );
		xVal = yVal = zVal = 1.0e-12;
	      }

	      vfield->set_value(Vector(xVal, yVal, zVal), node);

	      vector< double > errors;
	      errors.push_back( xVal );
	      errors.push_back( yVal );
	      errors.push_back( zVal );

	      efield->set_value(errors, node);
	    }
	  }

	  // Pressure
	  if( readOrder[PRESSURE] > -1 ) {

	    index = readOrder[PRESSURE];

	    pVal = data[index];

	    pfield->set_value(pVal, node);
	  }
	 
	  // Temperature
	  if( readOrder[TEMPERATURE] > -1 ) {

	    index = readOrder[TEMPERATURE];

	    tVal = data[index];

	    tfield->set_value(tVal, node);
	  }
	}
      }


      if( repeated ) {
	for( i=0; i<idim; i++ ) {
	  // Read the data in no matter the order.
	  for( index=0; index<nVals; index++ ) {
	    ifs >> data[index];			
	    
	    if( ifs.eof() ) {
	      error( string("Could not read grid ") + new_filename );
	    }
	  }
	}
      }

      // If needed repeat the radial values for this theta.
/*      if( !repeated )
      {
        StructHexVolMesh::Node::index_type node0;
	Point pt;

	node.j_ = jdim;
	node.k_ = k;

	node0.j_ = 0;
	node0.k_ = k;

	for( i=0; i<idim; i++ ) {

	  node.i_ = i;
	  node0.i_ = i;

	  hvm->get_center(pt, node0 );
	  hvm->set_point( pt, node );

	  if( bfield )
	    bfield->set_value(bfield->value( node0 ), node);
	  if( vfield )
	    vfield->set_value(vfield->value( node0 ), node);
	  if( pfield )
	    pfield->set_value(pfield->value( node0 ), node);
	  if( efield )
	    efield->set_value(efield->value( node0 ), node);
	}
      }
*/
    }

    if( repeated ) {
      for( j=0; j<jdim; j++ ) {
	for( i=0; i<idim; i++ ) {
	  // Read the data in no matter the order.
	  for( index=0; index<nVals; index++ ) {
	    ifs >> data[index];			

	    if( ifs.eof() ) {
	      error( string("Could not read grid ") + new_filename );
	    }
	  }
	}
      }
    }

/* Experiment for working in cylindrical coordinates.
    cerr << "min " << b_phi_min << "   max " << b_phi_max << endl;
*/
    // If needed repeat the first phi slice.
  /*    if( !repeated )
    {
      StructHexVolMesh::Node::index_type node0;
      Point pt;

      node.k_ = kdim;
      node0.k_ = 0;

      for( j=0; j<jdim+1; j++ ) {

	node.j_ = j;
	node0.j_ = j;

	for( i=0; i<idim; i++ ) {

	  node.i_ = i;
	  node0.i_ = i;

	  hvm->get_center(pt, node0 );
	  hvm->set_point( pt, node );

	  if( bfield )
	    bfield->set_value(bfield->value( node0 ), node);
	  if( vfield )
	    vfield->set_value(vfield->value( node0 ), node);
	  if( pfield )
	    pfield->set_value(pfield->value( node0 ), node);
	  if( efield )
	    efield->set_value(efield->value( node0 ), node);
	}
      }

    }
*/
    // Make sure that all of the data was read.
    string tmpStr;

    ifs >> tmpStr;

    if( !ifs.eof() ) {
      error( string("not all data was read ") + new_filename );
      error( string("not all data was read ") + tmpStr );
    }
  }
  else
  {
    remark( "Already read the file " +  new_filename );
  }


  // Get a handle to the output field port.
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

  // Get a handle to the output field port.
  if( tHandle_.get_rep() )
  {
    FieldOPort *ofield_port = 
      (FieldOPort *) get_oport("Output Temperature Field");

    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( tHandle_ );
  }

  // Get a handle to the output field port.
  if( bHandle_.get_rep() )
  {
    FieldOPort *ofield_port = 
      (FieldOPort *) get_oport("Output B Field");

    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( bHandle_ );
  }

  // Get a handle to the output field port.
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

  // Get a handle to the output field port.
  if( eHandle_.get_rep() )
  {
    FieldOPort *ofield_port = 
      (FieldOPort *) get_oport("Output Error Field");

    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( eHandle_ );
  }
}


} // End namespace Fusion




