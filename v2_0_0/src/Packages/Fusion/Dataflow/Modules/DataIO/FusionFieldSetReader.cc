//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : FusionFieldSetReader.cc
//    Author : Martin Cole
//    Date   : Thu Mar 20 13:01:41 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Teem/Core/Datatypes/NrrdData.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Core/Datatypes/StructHexVolField.h>

#include <Packages/Fusion/share/share.h>

#include <fstream>
#include <sys/stat.h>

namespace Fusion {

using namespace SCIRun;
using namespace SCITeem;

class FusionSHARE FusionFieldSetReader : public Module {
public:
  FusionFieldSetReader(GuiContext *context);

  virtual ~FusionFieldSetReader();

  virtual void execute();

private:
  void read_file(string newfilename, StructHexVolMesh *&hvm, 
		 double *&nrrd_data, ostringstream &tup, 
		 unsigned int &isz, unsigned int &jsz, unsigned int &ksz, 
		 unsigned int &d_vals, unsigned int &ntime,
		 unsigned int cur_time);

  GuiString filename_;
  string old_filename_;
  time_t old_filemodification_;

  NrrdDataHandle  handle_;
  NrrdOPort      *outport_;
};


DECLARE_MAKER(FusionFieldSetReader)


  FusionFieldSetReader::FusionFieldSetReader(GuiContext *context)
    : Module("FusionFieldSetReader", context, Source, "DataIO", "Fusion"),
      filename_(context->subVar("filename"))
{
}

FusionFieldSetReader::~FusionFieldSetReader(){
}

void 
FusionFieldSetReader::execute()
{
      
  string new_filename(filename_.get());
  
  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
  if (stat(new_filename.c_str(), &buf)) {
    error( string("File not found ") + new_filename );
    return;
  }
  

  double *nrrd_data = 0; // Will become the nrrd.
  ostringstream tup;
  StructHexVolMesh *hvm = NULL;
  unsigned int ksz = 0;
  unsigned int jsz = 0;
  unsigned int isz = 0;
  unsigned int d_vals = 0; 
  vector<string> fnames;
   

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


    // The series of names we need to read look like this:
    // NIMROD_10089_00.mds NIMROD_10089_01.mds NIMROD_10089_02.mds

    int fnumber = 0;
    unsigned int start_num = new_filename.size();
    unsigned int end_num = new_filename.size() - 4;

    string pre;
    string post;
    for (unsigned int i = end_num; i >= 0; i--) {
      if (new_filename[i] == '_') {
	// this index is the start of the number
	start_num = i;
	break;
      }
    }
    pre = new_filename.substr(0, start_num + 1);
    post = new_filename.substr(end_num, 4);
    string fnum = new_filename.substr(start_num + 1, end_num - start_num - 1);
    fnumber = atoi(fnum.c_str());
    //cout << "pre : " << pre << endl;
    //cout << "post: " << post << endl;
    //cout << "fnum string: " << fnum << endl;
    //cout << "fnumber: " << fnumber << endl;

    struct stat buf1;
    while (! stat(new_filename.c_str(), &buf1)) {
      fnames.push_back(new_filename);      
      ostringstream fname;
      fname << pre;
      fname.width(end_num - start_num - 1);
      fname.fill('0');
      fname << ++fnumber;
      fname << post;
      new_filename = fname.str();

    }
    vector<string>::iterator iter = fnames.begin();
    unsigned int cur_time = 0;
    unsigned int tsz = fnames.size();
    while (iter != fnames.end()) {
      remark("adding " + *iter + " to the series");
      read_file(*iter, hvm, nrrd_data, tup, 
		isz, jsz, ksz, d_vals, tsz, cur_time);
      ++iter; ++cur_time;
    }

  } else {
    remark( "Already read the file " +  new_filename );
  }

  NrrdData *nout = scinew NrrdData(1);
  nrrdWrap(nout->nrrd, nrrd_data, nrrdTypeDouble, 5, 
	   d_vals, isz, jsz, ksz, fnames.size());
  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, nrrdCenterNode,
		  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
  nout->nrrd->axis[0].label = strdup(tup.str().c_str());
  nout->nrrd->axis[1].label = strdup("i");
  nout->nrrd->axis[2].label = strdup("j");
  nout->nrrd->axis[3].label = strdup("k");
  nout->nrrd->axis[4].label = strdup("Time");


  // we need a field to store in the NrrdData so that the Teem
  // Package knows how to make a scirun field from the nrrd.
  StructHexVolField<double> *field = 
    scinew StructHexVolField<double>(hvm, Field::NODE);
    
  nout->set_orig_field(FieldHandle(field));

  handle_ = nout;
  outport_ = (NrrdOPort *)get_oport("Output Data");
  if (!outport_) {
    error("Unable to initialize oport 'Outport Data'.");
    return;
  }

  // Send the data downstream
  outport_->send(handle_);
}

void 
FusionFieldSetReader::read_file(string newfilename, StructHexVolMesh *&hvm, 
				double *&nrrd_data, ostringstream &tup, 
				unsigned int &isz, unsigned int &jsz, 
				unsigned int &ksz, 
				unsigned int &d_vals, unsigned int &ntime,
				unsigned int cur_time)
{
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



  // Read whether the data is wrapped our clamped
  bool repeated = false;
  unsigned int idim, jdim, kdim;
  int nVals;

  // Open the mesh file.
  ifstream ifs( newfilename.c_str() );

  if (!ifs) {
    error( string("Could not open file ") + newfilename );
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
      if (cur_time == 0) {
	tup << "BField:Vector";
	if (index < nVals) { tup << ","; }
      }
    }

    else if( strcmp( token, "BFIELD_R_PHI_Z" ) == 0 ) {
      readOrder[BFIELD_R_PHI_Z] = index;
      index += 3;
      if (cur_time == 0) {
	tup << "BField:Vector";
	if (index < nVals) { tup << ","; }
      }
    }

    else if( strcmp( token, "BFIELD_R_Z_PHI" ) == 0 ) {
      readOrder[BFIELD_R_Z_PHI] = index;
      index += 3;
      if (cur_time == 0) {
	tup << "BField:Vector";
	if (index < nVals) { tup << ","; }
      }
    }


    // Velocity
    else if( strcmp( token, "VFIELD_X_Y_Z" ) == 0 ) {
      readOrder[VFIELD_XYZ] = index;
      index += 3;
      if (cur_time == 0) {
	tup << "VField:Vector";
	if (index < nVals) { tup << ","; }
      }
    }

    else if( strcmp( token, "VFIELD_R_PHI_Z" ) == 0 ) {
      readOrder[VFIELD_R_PHI_Z] = index;
      index += 3;
      if (cur_time == 0) {
	tup << "VField:Vector";     
	if (index < nVals) { tup << ","; }
      }
    }

    else if( strcmp( token, "VFIELD_R_Z_PHI" ) == 0 ) {
      readOrder[VFIELD_R_Z_PHI] = index;
      index += 3;
      if (cur_time == 0) {
	tup << "VField:Vector";
	if (index < nVals) { tup << ","; }
      }
    }

    // Pressure
    else if( strcmp( token, "PRESSURE" ) == 0 ) {
      readOrder[PRESSURE] = index;
      index += 1;
      if (cur_time == 0) {
	tup << "Pressure:Scalar";
	if (index < nVals) { tup << ","; }     
      }
    }

    // Temperature
    else if( strcmp( token, "TEMPERATURE" ) == 0 ) {
      readOrder[TEMPERATURE] = index;
      index += 1;
      if (cur_time == 0) {
	tup << "Temperature:Scalar";
	if (index < nVals) { tup << ","; }
      }
    } else { 
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


  ifs >> token;

  if( strcmp( token, "REPEATED" ) == 0 )
    repeated = true;
  else if( strcmp( token, "SINGULAR" ) == 0 )
    repeated = false;
  else {
    error( string("Bad token: expected REPEATED or SINGULAR but read ") + token );
    return;
  }


  ifs >> idim >> jdim >> kdim; 

  //    remark( idim + "  " + jdim + "  " + kdim );

  // Create the grid, and scalar and vector data matrices.

  if( readOrder[GRID_XYZ    ] > -1 ||
      readOrder[GRID_R_PHI_Z] > -1 ||
      readOrder[GRID_R_Z_PHI] > -1 ) {
  
    if (!hvm) {
      if( repeated )
	hvm = scinew StructHexVolMesh(idim, jdim-1, kdim-1);
      else
	hvm = scinew StructHexVolMesh(idim, jdim, kdim);
    }
  }
  else {
    error( string("No grid present - unable to create the field(s).") );
    return;
  }
    
  // Read the data.
  double *data = new double[nVals];
      
  double xVal, yVal, zVal, pVal, tVal, rad, phi;

  bool phi_read;

  StructHexVolMesh::Node::index_type node;

  register unsigned int i, j, k;
  if(repeated) { 
    ksz = kdim - 1;
    jsz = jdim - 1;
  } else { 
    ksz = kdim;
    jsz = jdim;
  }
  isz = idim;
  d_vals = nVals - 3;

  if (!nrrd_data) {
    nrrd_data = new double[ntime * ksz * jsz * isz * d_vals]; 
  }

  unsigned toff = cur_time * ksz * jsz * isz * d_vals;
  unsigned koff, joff, ioff;
  for( k=0; k < ksz; k++ ) {
    koff = k * jsz * isz * d_vals;
    for( j=0; j < jsz; j++ ) {
      joff = j * isz * d_vals;
      for( i=0; i < isz; i++ ) {
	ioff = i * d_vals;
	// Read the data in no matter the order.
	for( index=0; index<nVals; index++ ) {
	  ifs >> data[index];			
	    
	  if( ifs.eof() ) {
	    error( string("Could not read grid ") + newfilename );
	    return;
	  }
	}

	phi_read = false;

	node.i_ = i;
	node.j_ = j;
	node.k_ = k;

	if (cur_time == 0) { // only fill the mesh once.
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

	    hvm->set_point(Point(xVal, yVal, zVal), node);
	  }
	}

	// B Field
	if( readOrder[BFIELD_XYZ] > -1 ) {

	  index = readOrder[BFIELD_XYZ];

	  xVal = data[index  ];
	  yVal = data[index+1];
	  zVal = data[index+2];

	  nrrd_data[toff + koff + joff + ioff + index - 3] = xVal;
	  nrrd_data[toff + koff + joff + ioff + index - 2] = yVal;
	  nrrd_data[toff + koff + joff + ioff + index - 1] = zVal;
	}
	else if( readOrder[BFIELD_R_PHI_Z] > -1 ) {

	  index = readOrder[BFIELD_R_PHI_Z];

	  if( phi_read ) {

	    xVal =  data[index  ] * cos(phi) -
	      data[index+1] * sin(phi);
	    yVal = -data[index  ] * sin(phi) -
	      data[index+1] * cos(phi );
	    zVal =  data[index+2];

	    nrrd_data[toff + koff + joff + ioff + index - 3] = xVal;
	    nrrd_data[toff + koff + joff + ioff + index - 2] = yVal;
	    nrrd_data[toff + koff + joff + ioff + index - 1] = zVal;
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

	    nrrd_data[toff + koff + joff + ioff + index - 3] = xVal;
	    nrrd_data[toff + koff + joff + ioff + index - 2] = yVal;
	    nrrd_data[toff + koff + joff + ioff + index - 1] = zVal;
	  }
	}

	// V Field
	if( readOrder[VFIELD_XYZ] > -1 ) {

	  index = readOrder[VFIELD_XYZ];

	  xVal = data[index  ];
	  yVal = data[index+1];
	  zVal = data[index+2];

	  nrrd_data[toff + koff + joff + ioff + index - 3] = xVal;
	  nrrd_data[toff + koff + joff + ioff + index - 2] = yVal;
	  nrrd_data[toff + koff + joff + ioff + index - 1] = zVal;
	}
	else if( readOrder[VFIELD_R_PHI_Z] > -1 ) {

	  index = readOrder[VFIELD_R_PHI_Z];


	  if( phi_read ) {

	    xVal =  data[index  ] * cos(phi) -
	      data[index+1] * sin(phi);
	    yVal = -data[index  ] * sin(phi) -
	      data[index+1] * cos(phi);
	    zVal =  data[index+2];
	      
	    nrrd_data[toff + koff + joff + ioff + index - 3] = xVal;
	    nrrd_data[toff + koff + joff + ioff + index - 2] = yVal;
	    nrrd_data[toff + koff + joff + ioff + index - 1] = zVal;
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

	    nrrd_data[toff + koff + joff + ioff + index - 3] = xVal;
	    nrrd_data[toff + koff + joff + ioff + index - 2] = yVal;
	    nrrd_data[toff + koff + joff + ioff + index - 1] = zVal;
	  }
	}

	// Pressure
	if( readOrder[PRESSURE] > -1 ) {

	  index = readOrder[PRESSURE];

	  pVal = data[index];

	  nrrd_data[toff + koff + joff + ioff + index - 3] = pVal;

	}
	 
	// Temperature
	if( readOrder[TEMPERATURE] > -1 ) {

	  index = readOrder[TEMPERATURE];

	  tVal = data[index];

	  nrrd_data[toff + koff + joff + ioff + index - 3] = tVal;
	}
      }
    }


    if( repeated ) {
      for( i=0; i<idim; i++ ) {
	// Read the data in no matter the order.
	for( index=0; index<nVals; index++ ) {
	  ifs >> data[index];			
	    
	  if( ifs.eof() ) {
	    error( string("Could not read grid ") + newfilename );
	  }
	}
      }
    }
  }

  if( repeated ) {
    for( j=0; j<jdim; j++ ) {
      for( i=0; i<idim; i++ ) {
	// Read the data in no matter the order.
	for( index=0; index<nVals; index++ ) {
	  ifs >> data[index];			

	  if( ifs.eof() ) {
	    error( string("Could not read grid ") + newfilename );
	  }
	}
      }
    }
  }


  // Make sure that all of the data was read.
  string tmpStr;
  
  ifs >> tmpStr;
  
  if( !ifs.eof() ) {
    error( string("not all data was read ") + newfilename );
    cerr << "forgot to read the following: " << endl;
    do {
      cerr << tmpStr << endl;
      ifs >> tmpStr;
    } while (!ifs.eof());
  }

}



} // End namespace Fusion




