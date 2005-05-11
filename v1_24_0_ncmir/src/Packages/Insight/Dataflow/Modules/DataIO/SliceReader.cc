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
 *  SliceReader.cc: Read in a Nrrd
 *
 *  Written by:
 *   Darby Van Uitert
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   April 2005
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Util/sci_system.h>
#include <Core/Containers/StringUtil.h>
#include <sys/stat.h>
#include <sstream>

#include <Core/Algorithms/DataIO/AnalyzeSliceImageIO.h>

#include "itkCastImageFilter.h"

//namespace SCITeem {
namespace Insight {

using namespace SCIRun;

class SliceReader : public Module {
public:
  SliceReader(SCIRun::GuiContext* ctx);
  virtual ~SliceReader();
  virtual void execute();
  virtual void tcl_command(GuiArgs& args, void* userdata);
private:
  GuiString       p_type_;
  GuiInt          size_0_;
  GuiInt          size_1_;
  GuiInt          size_2_;
  GuiDouble       spacing_0_;
  GuiDouble       spacing_1_;
  GuiDouble       spacing_2_;
  GuiInt          slice_;
  GuiInt          cast_output_;

  GuiFilename     filename_;

  //NrrdDataHandle  read_handle_;
  ITKDatatypeHandle  read_handle_;
  itk::AnalyzeSliceImageIO::Pointer io_; 
  FILE*           fp_;
  time_t          old_filemodification_;
  string          old_filename_;
  unsigned int    pixel_size_;

  //bool create_slice(NrrdData* nd);
  template< class data >
  bool create_slice(ITKDatatype* nd);
};

} // end namespace SCITeem

using namespace Insight;

DECLARE_MAKER(SliceReader)

SliceReader::SliceReader(SCIRun::GuiContext* ctx) : 
  Module("SliceReader", ctx, Filter, "DataIO", "Insight"),
  p_type_(ctx->subVar("p_type")),
  size_0_(ctx->subVar("size_0")),
  size_1_(ctx->subVar("size_1")),
  size_2_(ctx->subVar("size_2")),
  spacing_0_(ctx->subVar("spacing_0")),
  spacing_1_(ctx->subVar("spacing_1")),
  spacing_2_(ctx->subVar("spacing_2")),
  slice_(ctx->subVar("slice")),
  cast_output_(ctx->subVar("cast_output")),
  filename_(ctx->subVar("filename")),
  read_handle_(0),
  fp_(0),
  old_filemodification_(0),
  pixel_size_(0)
{
}

SliceReader::~SliceReader()
{
}


void
SliceReader::execute()
{

  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);

  slice_.reset();

  update_state(NeedData);

  // Read filename
  
  filename_.reset();
  string fn(filename_.get());
  if (fn == "") { 
    error("Please specify nrrd filename");
    return; 
  }

  // allow user to click on .img file also
  if (fn.substr(fn.length()-4,4) == ".img") {
    fn = fn.substr(0, fn.length()-4) + ".hdr";
    struct stat buf;
    if (stat(fn.c_str(), &buf) == - 1) {
      error("Error: Corresponding .hdr file does not exist.");
      return;
    }
  }

  // Read the status of this file so we can compare modification timestamps.
  struct stat buf;
  if (stat(fn.c_str(), &buf) == - 1) {
    error(string("SliceReader error - file not found: '")+fn+"'");
    return;
  }

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif

  if (fn != old_filename_ ||
      new_filemodification != old_filemodification_) {

    // new file
    old_filename_ = fn;
    old_filemodification_ = new_filemodification;
    
    // read header information
    io_ = itk::AnalyzeSliceImageIO::New();
    if (!io_->CanReadFile(fn.c_str())) {
      error("Cannot read " + fn + "!");
      return;
    }
    
    io_->SetFileName(fn.c_str());
    io_->ReadImageInformation();
    
    // update information

    // number of samples
    // Could be a 3D dataset or 4D with size 1 of time for last axis
    if (!(io_->GetNumberOfDimensions() == 3 || (io_->GetNumberOfDimensions() == 4 && io_->GetNumberOfPixels(3) == 1))) {
      error("SliceReader only handles 3 dimensional data\n");
      return;
    }
    size_0_.set(io_->GetNumberOfPixels(0));
    size_1_.set(io_->GetNumberOfPixels(1));
    size_2_.set(io_->GetNumberOfPixels(2));

    spacing_0_.set(io_->GetSpacing(0));
    spacing_1_.set(io_->GetSpacing(1));
    spacing_2_.set(io_->GetSpacing(2));

    if (slice_.get() >= size_2_.get()) 
      slice_.set(0);

    gui->execute(id + " configure_slice_slider " + to_string(size_2_.get()-1));
  }

  //NrrdData *nd = scinew NrrdData();
  ITKDatatype *nd = scinew ITKDatatype;

  // pixel type
  itk::ImageIOBase::IOComponentType p_type = io_->GetComponentType();
  switch(p_type) {
  case itk::ImageIOBase::CHAR:
    p_type_.set("CHAR");
    //    nrrdAlloc(nd->nrrd, nrrdTypeChar, 2, size_0_.get(), 
    //      size_1_.get());
    pixel_size_ = sizeof(char);

    // Read next slice
    if (!create_slice<char>(nd)) {
      error("Error reading slice: " + to_string(slice_.get()));
      return;
    } 

    break;
  case itk::ImageIOBase::UCHAR:
    p_type_.set("UCHAR");
    //    nrrdAlloc(nd->nrrd, nrrdTypeUChar, 2, size_0_.get(), 
    //      size_1_.get());
    pixel_size_ = sizeof(unsigned char);
    
    // Read next slice
    if (!create_slice<unsigned char>(nd)) {
      error("Error reading slice: " + to_string(slice_.get()));
      return;
    }
    break;
  case itk::ImageIOBase::SHORT:
    p_type_.set("SHORT");
    //    nrrdAlloc(nd->nrrd, nrrdTypeShort, 2, size_0_.get(), 
    //      size_1_.get());
    pixel_size_ = sizeof(short);

    // Read next slice
    if (!create_slice<short>(nd)) {
      error("Error reading slice: " + to_string(slice_.get()));
      return;
    }
    break;
  case itk::ImageIOBase::USHORT:
    p_type_.set("USHORT");
    //    nrrdAlloc(nd->nrrd, nrrdTypeUShort, 2, size_0_.get(), 
    //      size_1_.get());
    pixel_size_ = sizeof(unsigned short);

    // Read next slice
    if (!create_slice<unsigned short>(nd)) {
      error("Error reading slice: " + to_string(slice_.get()));
      return;
    }
    break;
  case itk::ImageIOBase::INT:
    p_type_.set("INT");
    //    nrrdAlloc(nd->nrrd, nrrdTypeInt, 2, size_0_.get(), 
    //      size_1_.get());
    pixel_size_ = sizeof(int);

    // Read next slice
    if (!create_slice<int>(nd)) {
      error("Error reading slice: " + to_string(slice_.get()));
      return;
    }
    break;
  case itk::ImageIOBase::UINT:
    p_type_.set("UINT");
    //    nrrdAlloc(nd->nrrd, nrrdTypeUInt, 2, size_0_.get(), 
    //      size_1_.get());
    pixel_size_ = sizeof(unsigned int);

    // Read next slice
    if (!create_slice<unsigned int>(nd)) {
      error("Error reading slice: " + to_string(slice_.get()));
      return;
    }
    break;
  case itk::ImageIOBase::FLOAT:
    p_type_.set("FLOAT");
    //    nrrdAlloc(nd->nrrd, nrrdTypeFloat, 2, size_0_.get(), 
    //      size_1_.get());
    pixel_size_ = sizeof(float);

    // Read next slice
    if (!create_slice<float>(nd)) {
      error("Error reading slice: " + to_string(slice_.get()));
      return;
    }
    break;
  case itk::ImageIOBase::DOUBLE:
    p_type_.set("DOUBLE");
    //    nrrdAlloc(nd->nrrd, nrrdTypeDouble, 2, size_0_.get(), 
    //      size_1_.get());
    pixel_size_ = sizeof(double);

    // Read next slice
    if (!create_slice<double>(nd)) {
      error("Error reading slice: " + to_string(slice_.get()));
      return;
    }
    break;
  default:
    p_type_.set("UNKNOWN");
    error("Error: Unknown Analyze data type.");
    return;
    break;
  }

  // Send the data downstream.
  //NrrdDataOPort *outport = (NrrdDataOPort *)get_oport("OutputSlice");
  ITKDatatypeOPort *outport = (ITKDatatypeOPort *)get_oport("OutputSlice");
  outport->send(read_handle_);

  update_state(Completed);
}

//SliceReader::create_slice(NrrdData* nd)
template<class data>
bool
SliceReader::create_slice(ITKDatatype* nd)
{

  io_->CloseImageFile(fp_);
  fp_ = io_->OpenImageFile(io_->GetImageFile(old_filename_));
  
  // set spacing, min and max
//   nd->nrrd->axis[0].spacing = io_->GetSpacing(0);
//   nd->nrrd->axis[1].spacing = io_->GetSpacing(1);
//   nd->nrrd->axis[0].min = io_->GetOrigin(0);
//   nd->nrrd->axis[1].min = io_->GetOrigin(1);
//   nrrdAxisInfoMinMaxSet(nd->nrrd, 0, nrrdCenterNode); 
//   nrrdAxisInfoMinMaxSet(nd->nrrd, 1, nrrdCenterNode); 
  
  if (!fp_) 
    return false;

  // create the image, set the origin and spacing
  typedef typename itk::Image<data,2> ImageType;
  typename ImageType::Pointer img = ImageType::New();
  typename ImageType::SizeType fixedSize = {{size_0_.get(), size_1_.get()}};
  img->SetRegions( fixedSize );

  double origin[2];
  origin[0] = io_->GetOrigin(0);
  origin[1] = io_->GetOrigin(1);
  img->SetOrigin( origin );

  double spacing[2];
  spacing[0] = io_->GetSpacing(0);
  spacing[1] = io_->GetSpacing(1);
  img->SetSpacing( spacing );

  img->Allocate();
  data* d = img->GetPixelContainer()->GetImportPointer();

  // read the current slice
  int slice = slice_.get();

  if (slice >= size_2_.get()) {
    error("Error: Attempting to read slice beyond bounds of data.");
    return false;
  }
  
  long offset = size_0_.get() * size_1_.get() * pixel_size_ * slice;
  fseek(fp_, offset, SEEK_SET);
  if(!fread(d, pixel_size_, 
 	    size_0_.get() * size_1_.get(), fp_)) {
    error("Error: Couldn't read next slice.");
    return false;
  }

  // cast if indicated
  if (cast_output_.get() == 1) {
    typedef typename itk::Image<float, 2> CastType;
    typedef typename itk::CastImageFilter< ImageType, CastType> CastImageFilterType;
    typename CastImageFilterType::Pointer caster = CastImageFilterType::New();

    caster->SetInput(img);

    // execute the caster
    try {
      caster->Update();
    } catch ( itk::ExceptionObject & err ) {
      error("ExceptionObject caught!");
      error(err.GetDescription());
    }
    nd->data_ = caster->GetOutput();
  } else {
    nd->data_ = img;
  }

  read_handle_ = nd;
  return true;
}


void 
SliceReader::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2)
  {
    args.error("SliceReader needs a minor command");
    return;
  }
  Module::tcl_command(args, userdata);
}
