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
 *  VULCANDataReader.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   July 2003
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/share/share.h>

#include <Dataflow/Ports/FieldPort.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>

#include <sci_defs.h>

#include <sys/stat.h>

#include <fstream>

namespace Fusion {

using namespace SCIRun;
using namespace SCITeem;

class PSECORESHARE VULCANDataReader : public Module {
public:
  VULCANDataReader(GuiContext*);

  virtual ~VULCANDataReader();

  virtual void tcl_command(GuiArgs&, void*);

  virtual void execute();

private:
  GuiFilename filename_;

  NrrdDataHandle nHandles_[9];
  MatrixHandle mHandle_;

  string old_filename_;
  time_t old_filemodification_;

  bool error_;
};


DECLARE_MAKER(VULCANDataReader)
VULCANDataReader::VULCANDataReader(GuiContext* context)
  : Module("VULCANDataReader", context, Source, "DataIO", "Fusion"),
    filename_(context->subVar("filename")),
    error_(false)
{
}

VULCANDataReader::~VULCANDataReader(){
}

void
VULCANDataReader::execute(){

  string portNames[10] = { "Grid Points",
			   "Grid Vector",
			   "Cell Connections",
			   "Cell Density",
			   "Cell Temperature",
			   "Cell Pressure",
			   "Cell Ye Fraction",
			   "Cell Entropy",
			   "Cell Angular Velocity",
			   "Time Slice" };

  string new_filename(filename_.get());

  if( new_filename.length() == 0 )
    return;

  // Read the status of this file so we can compare modification timestamps
#ifdef HAVE_STAT64
  struct stat64 buf;
  if (stat64(new_filename.c_str(), &buf)) {
#else
  struct stat buf;
  if (stat(new_filename.c_str(), &buf)) {
#endif
    error( string("Execute File not found ") + new_filename );
    return;
  }

  //  cerr << "\nnew " << new_filename << "\nold " << old_filename_ << endl;

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif

  if( error_ ||
      new_filename         != old_filename_ || 
      new_filemodification != old_filemodification_  ) {

    error_ = false;

    old_filemodification_ = new_filemodification;
    old_filename_         = new_filename;

    bool list;

    char base_filename[256];
    int start = 0, stop = 1, incr = 1, delay;
    string path;

    if( new_filename.find(".list") != std::string::npos ) {
      FILE *fp;
      
      if (!(fp = fopen(new_filename.c_str(), "r"))) {
	error("Unable to open " + new_filename);
	return;
      }

      fscanf(fp, "%s %d %d %d %d", base_filename, &start, &stop, &incr, &delay);

      fclose( fp );

      path = new_filename;

      std::string::size_type last = new_filename.find_last_of( "/" ) + 1;
      path.erase( last, path.length()-last);

      list = true;

    } else {
      start = 0;
      stop = 1;
      incr = 1;
      delay = 0;
      strcpy( base_filename, new_filename.c_str() );

      list = false;
    }


    for( int index=start; index<stop; index+=incr ) {

      bool last;

      if( index+incr<stop )
	last = false;
      else
	last =  true;

      if( list ) {
	char indexStr[8];

	sprintf( indexStr, ".%04d", index );

	new_filename = path + string( base_filename ) + indexStr;
      }

      int sink_size = 1;
      int ndims = 1;

      FILE *fp;

      /* Check to see that the file is accessible */
      if (!(fp = fopen(new_filename.c_str(), "r"))) {
	error("Unable to open " + new_filename);
	return;
      }

      remark( "Reading " + new_filename );

      char tstr[256];
      int junk1, junk2;
      float time;

      // Get the time value from the first line of the file.
      fscanf(fp, "%s %d %g %d\n", tstr, &junk1, &time, &junk2);

      if( ferror( fp ) || feof( fp ) ) {
	error("Unable to read " + new_filename);
	error_ = true;
	return;
      }

      // Time 
      ColumnMatrix *selected = scinew ColumnMatrix(1);
      selected->put(0, 0, (double) time);
      mHandle_ = MatrixHandle(selected);


      /* Throw away the next 7 lines of the file. */
      for(int i=0; i<8; i++)
	fgets(tstr, 256, fp);

      if( ferror( fp ) || feof( fp ) ) {
	error("Unable to read " + new_filename);
	error_ = true;
	return;
      }
    
      /* Read the number of positions from the file. */
      int npos;

      fscanf(fp, "%d", &npos);

      if( ferror( fp ) || feof( fp ) ) {
	error("Unable to read " + new_filename);
	error_ = true;
	return;
      }

      double*  pdata = scinew double[npos*3];
      double* vdata = scinew double[npos*3];
 
      double  *pptr = pdata;
      double *vptr = vdata;
 
      float z, r;
      double vz, vr;

      // Node Data
      for (int i=0; i<npos; i++) {
	fscanf(fp, "%g %g %lg %lg", &z, &r, &vz, &vr);
	*pptr++ = z;
	*pptr++ = r;
	*pptr++ = 0.0;
	*vptr++ = vz;
	*vptr++ = vr;
	*vptr++ = 0.0;

	if( ferror( fp ) || feof( fp ) ) {
	  error("Unable to read " + new_filename);
	  error_ = true;
	  return;
	}
      }

      // Points
      NrrdData *nout = scinew NrrdData(false);

      nrrdWrap(nout->nrrd, pdata, nrrdTypeDouble, ndims+1, sink_size, npos);

      nout->nrrd->axis[0].label = strdup("ZR:Vector");
      nout->nrrd->axis[1].label = strdup("Domain");
    
      nout->set_property( "Topology",          string("Unstructured"), false );
      nout->set_property( "Coordinate System", string("Cylindrical - VULCAN"), false );
    
      nHandles_[0] = NrrdDataHandle( nout );	

      // Vector
      nout = scinew NrrdData(false);

      nrrdWrap(nout->nrrd, vdata, nrrdTypeDouble, ndims+1, sink_size, npos);

      nout->nrrd->axis[0].label = strdup("ZR:Vector");
      nout->nrrd->axis[1].label = strdup("Domain");
    
      nout->set_property( "DataSpace",         string("REALSPACE"), false );
      nout->set_property( "Coordinate System", string("Cylindrical - VULCAN"), false );
    
      nHandles_[1] = NrrdDataHandle( nout );	


      // Cell Data
      int ncon;

      fscanf(fp, "%d", &ncon);
    
      if( ferror( fp ) || feof( fp ) ) {
	error("Unable to read " + new_filename);
	error_ = true;
	return;
      }

      double* cdata = scinew double[ncon*4];

      double *data[6];

      int c0, c1, c2, c3;

      for( int i=0; i<6; i++ )
	data[i] = scinew double[ncon];

      for( int i=0, j=0; i<ncon; i++, j+=4 ) {
	fscanf(fp, "%d %d %d %d %d %lg %lg %lg %lg %lg %lg",
	       &c0, &c1, &c2, &c3, &junk1,
	       &data[0][i], &data[1][i], &data[2][i],
	       &data[3][i], &data[4][i], &data[5][i]);

	cdata[j+0] = c0-1; 
	cdata[j+1] = c1-1;
	cdata[j+2] = c2-1;
	cdata[j+3] = c3-1;

	if( ferror( fp ) || feof( fp ) ) {
	  error("Unable to read " + new_filename);
	  error_ = true;
	  return;
	}
      }
      // Connections
      nout = scinew NrrdData(false);

      nrrdWrap(nout->nrrd, cdata, nrrdTypeDouble, ndims+2, sink_size, ncon, 4);

      nout->nrrd->axis[0].label = strdup("Connections:Scalar");
      nout->nrrd->axis[1].label = strdup("Domain");
      nout->nrrd->axis[2].label = strdup("Connections");
    
      nout->set_property( "Topology",  string("Unstructured"), false );
      nout->set_property( "Cell Type", string("Quad"), false );
    
      nHandles_[2] = NrrdDataHandle( nout );

      // Data
      for( int i=0; i<6; i++ ) {
	nout = scinew NrrdData(false);

	nrrdWrap(nout->nrrd, data[i], nrrdTypeDouble, ndims+1, sink_size, ncon);

	nout->nrrd->axis[0].label = strdup("Data:Scalar");
	nout->nrrd->axis[1].label = strdup("Domain");
    
	nout->set_property( "DataSpace",         string("REALSPACE"), false );
	nout->set_property( "Coordinate System", string("Cylindrical - VULCAN"), false );
    
	nHandles_[3+i] = NrrdDataHandle( nout );	
      }

      // Ignore everything else in the file.
      fclose(fp);

      if( !last ) {
	remark( "Sending itermediate." );

	for( int i=0; i<9; i++ ) {
	  // Get a handle to the output field port.
	  if( nHandles_[i].get_rep() ) {
	    NrrdOPort *ofield_port = (NrrdOPort *) get_oport(portNames[i]);
    
	    if (!ofield_port) {
	      error("Unable to initialize "+name+"'s oport" + portNames[i]+ "\n");
	      return;
	    }

	    // Send the data downstream
	    ofield_port->send_intermediate( nHandles_[i] );
	  }
	}

	if( mHandle_.get_rep() ) {
	  MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Time Slice");
    
	  if (!omatrix_port) {
	    error("Unable to initialize oport 'Time Slice'.");
	    return;
	  }
    
	  omatrix_port->send_intermediate(mHandle_);
	}

	if ( delay > 0) {
	  remark( "Delaying ... " );
	  const unsigned int secs = delay / 1000;
	  const unsigned int msecs = delay % 1000;
	  if (secs)  { sleep(secs); }
	  if (msecs) { usleep(msecs * 1000); }
	}
      } else if( list )
	remark( "Sending last." );
    }
  }


  for( int i=0; i<9; i++ ) {
    // Get a handle to the output field port.
    if( nHandles_[i].get_rep() ) {
      NrrdOPort *ofield_port = (NrrdOPort *) get_oport(portNames[i]);
    
      if (!ofield_port) {
	error("Unable to initialize "+name+"'s oport" + portNames[i]+ "\n");
	return;
      }

      // Send the data downstream
      ofield_port->send( nHandles_[i] );
    }
  }

  if( mHandle_.get_rep() ) {
    MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Time Slice");
    
    if (!omatrix_port) {
      error("Unable to initialize oport 'Time Slice'.");
      return;
    }
    
    omatrix_port->send(mHandle_);
  }
}

void
VULCANDataReader::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


}
