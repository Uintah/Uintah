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
 *  UnuSave.cc 
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>

#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuSave : public Module {
public:
  UnuSave(GuiContext*);

  virtual ~UnuSave();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  NrrdIPort*      inrrd_;

  GuiString       format_;
  GuiString       encoding_;
  GuiString       endian_;
  GuiString       filename_;
};


DECLARE_MAKER(UnuSave)
UnuSave::UnuSave(GuiContext* ctx)
  : Module("UnuSave", ctx, Source, "UnuNtoZ", "Teem"),
    inrrd_(0), 
    format_(ctx->subVar("format")),
    encoding_(ctx->subVar("encoding")),
    endian_(ctx->subVar("endian")),
    filename_(ctx->subVar("filename"))
{
}

UnuSave::~UnuSave(){
}

void
 UnuSave::execute(){
  NrrdDataHandle nrrd_handle;

  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");

  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty InputNrrd.");
    return;
  }


  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;

  NrrdIoState *nio = nrrdIoStateNew();

  string encoding = encoding_.get();
  if (encoding == "raw") 
    nio->encoding = nrrdEncodingArray[1];
  else if (encoding == "ascii") 
    nio->encoding = nrrdEncodingArray[2];
  else if (encoding == "hex")
    nio->encoding = nrrdEncodingArray[3];
  else {
    warning("Unkown encoding. Using raw.");
    nio->encoding = nrrdEncodingArray[1];
  }

  string format = format_.get();
  if (format == "nrrd")
    nio->format = nrrdFormatArray[1];
  else if (format == "pnm")
    nio->format = nrrdFormatArray[2];
  else if (format == "png")
    nio->format = nrrdFormatArray[3];
  else if (format == "vtk")
    nio->format = nrrdFormatArray[4];
  else if (format == "text")
    nio->format = nrrdFormatArray[5];
  else if (format == "eps")
    nio->format = nrrdFormatArray[6];
  else {
    warning("Unknown format.  Using nrrd.");
    nio->format = nrrdFormatArray[1];
  }

  string endian = endian_.get();
  if (endian == "little")
    nio->endian = airEndianLittle;
  else if (endian == "big")
    nio->endian = airEndianBig;
  else {
    warning("Unkown endiannes.  Using default of machine");
    nio->endian = airMyEndian;
  }

  if (AIR_ENDIAN != nio->endian) {
    nrrdSwapEndian(nin);
  }
  if (airEndsWith(filename_.get().c_str(), NRRD_EXT_NHDR)) {
    if (nio->format != nrrdFormatNRRD) {
      nio->format = nrrdFormatNRRD;
    }
  } 

  if (nrrdSave(filename_.get().c_str(), nin, nio)) {
    char *err = biffGet(NRRD);      
    cerr << "Error writing nrrd " << filename_.get() << ": "<< err << endl;
    free(err);
    biffDone(NRRD);
    return;
  }

}

void
 UnuSave::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


