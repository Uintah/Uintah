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
 *  NrrdWriter.cc: Nrrd writer
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <sstream>
#include <fstream>
using std::ifstream;
using std::ostringstream;

namespace SCITeem {

using namespace SCIRun;

class NrrdWriter : public Module {
protected:
    GuiFilename gui_filename_;
    GuiString   gui_filetype_;
    GuiString   gui_exporttype_;
public:
    NrrdWriter(GuiContext *ctx);
    virtual ~NrrdWriter();
    virtual void execute();
};

} // end namespace SCITeem

using namespace SCITeem;
DECLARE_MAKER(NrrdWriter)

NrrdWriter::NrrdWriter(GuiContext *ctx)
: Module("NrrdWriter", ctx, Filter, "DataIO", "Teem"), 
  gui_filename_(get_ctx()->subVar("filename")),
  gui_filetype_(get_ctx()->subVar("filetype")),
  gui_exporttype_(get_ctx()->subVar("exporttype"))
{
}

NrrdWriter::~NrrdWriter()
{
}

void NrrdWriter::execute()
{
  // Read data from the input port
  NrrdDataHandle nrrd_input_handle;

  if( !get_input_handle( "Input Data", nrrd_input_handle, true ) )
    return;

  // If there is a current index matrix, use it as part of the name.
  MatrixHandle matrix_input_handle = 0;

  if( get_input_handle( "Current Index", matrix_input_handle, false ) ) {
  }

  // If no data or a changed recalcute.
  if( inputs_changed_ ||
      gui_filename_.changed( true ) ||
      gui_filetype_.changed() ||
      gui_exporttype_.changed() ) {

    update_state( Executing );

    // If no name is provided, return
    string fn(gui_filename_.get());
    if(fn == "") {
      error("No filename in specified");
      return;
    }

    // get root of filename (no extension)
    string::size_type e = fn.find_last_of(".");
    string root = fn;
    if (e != string::npos) root = fn.substr(0,e);

    if( matrix_input_handle.get_rep() ) {
      int index = (int) (matrix_input_handle->get(0, 0));

      char tmp_string[8];

      sprintf( tmp_string, "_%04d", index );

      root += string(tmp_string);
    }

    // determine which case we are writing out based on the
    // original filename.  
    bool writing_nrrd = false;
    bool writing_nhdr = false;
    bool writing_nd = false;
    
    if (fn.find(".nrrd",0)      != string::npos) writing_nrrd = true;
    else if (fn.find(".nhdr",0) != string::npos) writing_nhdr = true;
    else if (fn.find(".nd",0)   != string::npos) writing_nd = true;

    // If the filename doesn't have an extension
    // use the export type to determine what it should be
    if (!writing_nrrd && !writing_nhdr && !writing_nd) {
      string type = gui_exporttype_.get();

      if (type.find(".nrrd",0) != string::npos)
	writing_nrrd = true;
      else writing_nhdr = true;
    }

    // only write out the .nd extension if that was the filename
    // specified or if there are properties.  In the case that it
    // was an .nd filename specified, write out a .nrrd file also.
    if (nrrd_input_handle->nproperties() > 0) {
      if (!writing_nd) {
	writing_nd = true;
	if (!writing_nhdr)
	  writing_nrrd = true;
      }
    }

    // writing out NrrdData - use Piostream
    if (writing_nd) {
      string nrrd_data_fn = root + ".nd";
      Piostream* stream;
      string ft(gui_filetype_.get());
      
      // set NrrdData's write_nrrd variable to indicate
      // whether NrrdData's io method should write out a .nrrd or .nhdr
      if (writing_nhdr) nrrd_input_handle.get_rep()->write_nrrd_ = false;
      else nrrd_input_handle.get_rep()->write_nrrd_ = true;

      if (ft == "Binary")
	  stream = scinew BinaryPiostream(nrrd_data_fn, Piostream::Write);
      else
	  stream = scinew TextPiostream(nrrd_data_fn, Piostream::Write);
    
      if (stream->error()) {
	error("Could not open file for writing" + nrrd_data_fn);
      } else {
	// Write the file
	Pio(*stream, nrrd_input_handle); // will also write out a separate nrrd.
	delete stream; 
      } 
    } else {
      // writing out ordinary .nrrd .nhdr file
      string nrrd_fn = root;
      if (writing_nhdr) nrrd_fn += ".nhdr";
      else nrrd_fn += ".nrrd";

      Nrrd *nin = nrrd_input_handle->nrrd_;
    
      NrrdIoState *nio = nrrdIoStateNew();
      // set encoding to be raw
      nio->encoding = nrrdEncodingArray[1];
      // set format to be nrrd
      nio->format = nrrdFormatArray[1];
      // set endian to be endian of machine
      nio->endian = airMyEndian;
    
      if (AIR_ENDIAN != nio->endian) {
	nrrdSwapEndian(nin);
      }
      if (writing_nhdr) {
	if (nio->format != nrrdFormatNRRD) {
	  nio->format = nrrdFormatNRRD;
	}
      }
    
      if (nrrdSave(nrrd_fn.c_str(), nin, nio)) {
	char *err = biffGet(NRRD);      
	cerr << "Error writing nrrd " << nrrd_fn << ": "<< err << endl;
	free(err);
	biffDone(NRRD);
	return;
      }
    }
  }
}

      
