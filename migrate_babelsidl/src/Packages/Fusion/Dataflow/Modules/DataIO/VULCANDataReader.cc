/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#include <sci_defs/stat64_defs.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/NrrdPort.h>

#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

#include <sys/stat.h>

#include <fstream>

namespace Fusion {

using namespace SCIRun;

class VULCANDataReader : public Module {
public:
  VULCANDataReader(GuiContext*);

  virtual ~VULCANDataReader();

  virtual void tcl_command(GuiArgs&, void*);

  virtual void execute();

private:
  GuiFilename gui_filename_;

  NrrdDataHandle nrrd_output_handles_[9];
  MatrixHandle matrix_output_handle_;

  string old_filename_;
  time_t old_filemodification_;

  bool loop_;
  int  index_;

  bool execute_error_;
};


DECLARE_MAKER(VULCANDataReader)
VULCANDataReader::VULCANDataReader(GuiContext* context)
  : Module("VULCANDataReader", context, Source, "DataIO", "Fusion"),
    gui_filename_(context->subVar("filename")),
    matrix_output_handle_(0),
    loop_(false),
    index_(0),
    execute_error_(false)
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

  string new_filename(gui_filename_.get());

  if( new_filename.length() == 0 )
    return;

  // Read the status of this file so we can compare modification timestamps
#ifdef HAVE_STAT64
  struct stat64 buf;
  if (stat64(new_filename.c_str(), &buf) == -1) {
#else
  struct stat buf;
  if (stat(new_filename.c_str(), &buf) == -1) {
#endif
    error( string("Execute File not found ") + new_filename );
    return;
  }

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif

  int delay = 0;

  if( loop_ ||
      execute_error_ ||
      new_filename         != old_filename_ || 
      new_filemodification != old_filemodification_  ) {

    update_state( Executing );

    execute_error_ = false;

    old_filemodification_ = new_filemodification;
    old_filename_         = new_filename;

    char base_filename[256];
    int start = 0, stop = 1, incr = 1;
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

      std::string::size_type pos = new_filename.find_last_of( "/" ) + 1;
      path.erase( pos, path.length()-pos);

      if( !loop_ ) {
	loop_ = true;
	index_ = start;
      } else {
	index_ += incr;
      }

      if( index_ + incr > stop )
	loop_ = false;

      char indexStr[8];
      
      sprintf( indexStr, ".%04d", index_ );
      
      new_filename = path + string( base_filename ) + indexStr;

    } else {
      strcpy( base_filename, new_filename.c_str() );
    }

    int rank = 3;
    int ndims = 1;

    FILE *fp;

    /* Check to see that the file is accessible */
    if (!(fp = fopen(new_filename.c_str(), "r"))) {
      error("Unable to open " + new_filename);
      execute_error_ = true;
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
      execute_error_ = true;
      return;
    }

    // Time 
    ColumnMatrix *selected = scinew ColumnMatrix(1);
    selected->put(0, 0, (double) time);
    matrix_output_handle_ = MatrixHandle(selected);


    /* Throw away the next 7 lines of the file. */
    for(int i=0; i<8; i++)
      fgets(tstr, 256, fp);

    if( ferror( fp ) || feof( fp ) ) {
      error("Unable to read " + new_filename);
      execute_error_ = true;
      return;
    }
    
    /* Read the number of positions from the file. */
    int npos;

    fscanf(fp, "%d", &npos);

    if( ferror( fp ) || feof( fp ) ) {
      error("Unable to read " + new_filename);
      execute_error_ = true;
      return;
    }

    double* pdata = scinew double[npos*3];
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
	execute_error_ = true;
	return;
      }
    }

    string nrrdName;

    // Points
    NrrdData *nout = scinew NrrdData();
    size_t size[NRRD_DIM_MAX];
    size[0] = rank;
    size[1] = npos;
    nrrdWrap_nva(nout->nrrd_, pdata, nrrdTypeDouble, ndims+1, size);

    nout->nrrd_->axis[0].kind  = nrrdKind3Vector;
    nout->nrrd_->axis[0].label = airStrdup("ZR");
    nout->nrrd_->axis[1].label = airStrdup("Domain");

    unsigned int centers[NRRD_DIM_MAX];
    centers[0] = nrrdCenterNode; centers[1] = nrrdCenterNode;
    centers[2] = nrrdCenterNode;
    nrrdAxisInfoSet_nva(nout->nrrd_, nrrdAxisInfoCenter, centers);

    nout->set_property( "Topology",          string("Unstructured"), false );
    nout->set_property( "Coordinate System", string("Cylindrical - VULCAN"), false );

    {
      ostringstream str;
      str << "Time-" << time << "-Points-ZR:Vector";

      nrrdName = string(str.str());

      nout->set_property( "Name", nrrdName, false );
    }

    nrrd_output_handles_[0] = NrrdDataHandle( nout );

    // Velocity Vector
    nout = scinew NrrdData();

    nrrdWrap_nva(nout->nrrd_, vdata, nrrdTypeDouble, ndims+1, size);

    nout->nrrd_->axis[0].kind  = nrrdKind3Vector;
    nout->nrrd_->axis[0].label = airStrdup("ZR");
    nout->nrrd_->axis[1].label = airStrdup("Domain");
    
    nrrdAxisInfoSet_nva(nout->nrrd_, nrrdAxisInfoCenter, centers);

    nout->set_property( "DataSpace",         string("REALSPACE"), false );
    nout->set_property( "Coordinate System", string("Cylindrical - VULCAN"), false );
    
    {
      ostringstream str;
      str << "Time-" << time << "-Veclocity-ZR:Vector";

      nrrdName = string(str.str());

      nout->set_property( "Name", nrrdName, false );
    }

    nrrd_output_handles_[1] = NrrdDataHandle( nout );


    // Cell Data
    int ncon;

    fscanf(fp, "%d", &ncon);
    
    if( ferror( fp ) || feof( fp ) ) {
      error("Unable to read " + new_filename);
      execute_error_ = true;
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
	execute_error_ = true;
	return;
      }
    }

    // Connections
    nout = scinew NrrdData();

    size_t con_size[NRRD_DIM_MAX];
    con_size[0] = 4;
    con_size[1] = ncon;
    nrrdWrap_nva(nout->nrrd_, cdata, nrrdTypeDouble, ndims+1, con_size);

    nout->nrrd_->axis[0].label = airStrdup("Connections");
    nout->nrrd_->axis[1].label = airStrdup("Domain");
    
    nrrdAxisInfoSet_nva(nout->nrrd_, nrrdAxisInfoCenter, centers);

    nout->set_property( "Topology",  string("Unstructured"), false );
    nout->set_property( "Cell Type", string("Quad"), false );
    
    {
      ostringstream str;
      str << "Time-" << time << "-Connections-Quad:Scalar";

      nrrdName = string(str.str());

      nout->set_property( "Name", nrrdName, false );
    }

    nrrd_output_handles_[2] = NrrdDataHandle( nout );

    // Data
    for( int i=0; i<6; i++ ) {
      size_t data_size[NRRD_DIM_MAX];
      data_size[0] = ncon;

      nout = scinew NrrdData();

      nrrdWrap_nva(nout->nrrd_, data[i], nrrdTypeDouble, ndims, data_size);

      nout->nrrd_->axis[0].label = airStrdup("Domain");
    
      nout->set_property( "DataSpace",         string("REALSPACE"), false );
      nout->set_property( "Coordinate System", string("Cylindrical - VULCAN"), false );
    
      {
	ostringstream str;
	str << "Time-" << time << "-Data-" << portNames[i] << ":Scalar";

	nrrdName = string(str.str());

	nout->set_property( "Name", nrrdName, false );
      }
 
      nrrd_output_handles_[3+i] = NrrdDataHandle( nout );	
    }

    // Ignore everything else in the file.
    fclose(fp);
  }

  if( loop_ )
    remark( "Sending itermediate." );
  else
    remark( "Sending last." );

  for( int i=0; i<9; i++ ) {

    // Get a handle to the output field port.
    if( nrrd_output_handles_[i].get_rep() ) {
      nrrd_output_handles_[i]->set_property( "Source",
					     string("Vulcan Reader"),
					     false );

      // Send the data downstream
      send_output_handle( portNames[i], nrrd_output_handles_[i], true );
    }
  }

  send_output_handle( "Time Slice", matrix_output_handle_, true );

  if( loop_ ) {
    if ( delay > 0) {
      remark( "Delaying ... " );
      const unsigned int secs = delay / 1000;
      const unsigned int msecs = delay % 1000;
      if (secs)  { sleep(secs); }
      if (msecs) { usleep(msecs * 1000); }
    }
    
    remark( "Executing. " );
    want_to_execute();
  }
}

void
VULCANDataReader::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


}
