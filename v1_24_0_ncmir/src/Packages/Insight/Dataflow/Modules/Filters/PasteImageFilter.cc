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
 *  PasteImageFilter.cc:
 *
 *  Written by:
 *   Darby Van Uitert
 *   January 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/share/share.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>


namespace Insight {

using namespace SCIRun;

class PSECORESHARE PasteImageFilter : public Module {
public:
  PasteImageFilter(GuiContext*);

  virtual ~PasteImageFilter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiInt                 size0_;
  GuiInt                 size1_;
  GuiInt                 size2_;
  GuiDouble              fill_value_;
  GuiInt                 axis_;
  GuiInt                 index_;
  ITKDatatypeIPort*      img_;
  ITKDatatypeOPort*      oimg_;
  int                    generation_;
  ITKDatatypeHandle      imgH_;
  ITKDatatypeHandle      volH_;
  int                    mode_;
  int                    old_mode_;

  void generate_volume();
};


DECLARE_MAKER(PasteImageFilter)
PasteImageFilter::PasteImageFilter(GuiContext* ctx)
  : Module("PasteImageFilter", ctx, Source, "Filters", "Insight"),
    size0_(ctx->subVar("size0")), size1_(ctx->subVar("size1")),
    size2_(ctx->subVar("size2")), fill_value_(ctx->subVar("fill_value")),
    axis_(ctx->subVar("axis")), index_(ctx->subVar("index")), 
    img_(0), oimg_(0), generation_(-1),
    imgH_(0), volH_(0), mode_(0), old_mode_(-1)
{
}

PasteImageFilter::~PasteImageFilter(){
}


void PasteImageFilter::execute(){
  
  update_state(NeedData);
  img_ = (ITKDatatypeIPort *)get_iport("SubImage");
  oimg_ = (ITKDatatypeOPort *)get_oport("OutputImage");
  
  if (!img_) {
    error("Unable to initialize iport 'SubImage'.");
    return;
  }
  if (!oimg_) {
    error("Unable to initialize oport 'OutputImage'.");
    return;
  }
  
  img_->get(imgH_);
  if (!imgH_.get_rep()) {
    error("Empty Sub Image.");
    return;
  }

  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);

  typedef itk::Image<unsigned char,2> SliceImageTypeUChar;
  typedef itk::Image<unsigned char,3> VolumeImageTypeUChar;

  typedef itk::Image<float,2> SliceImageTypeFloat;
  typedef itk::Image<float,3> VolumeImageTypeFloat;

  if (!dynamic_cast<SliceImageTypeUChar* >(imgH_.get_rep()->data_.GetPointer()) &&
      !dynamic_cast<SliceImageTypeFloat* >(imgH_.get_rep()->data_.GetPointer())) {
    error("PasteImageFilter module currently only supports float or unsigned char datatypes.");
    return;
  }

  if (dynamic_cast<SliceImageTypeUChar* >(imgH_.get_rep()->data_.GetPointer())) {
    mode_ = 0;
  } else {
    mode_ = 1;
  }

  if (generation_ == -1 || old_mode_ != mode_) {
    // first time through
    generate_volume();
  }

  SliceImageTypeUChar *slice0 = NULL;
  SliceImageTypeFloat *slice1 = NULL;
  VolumeImageTypeUChar *volume0 = NULL;
  VolumeImageTypeFloat *volume1 = NULL;
  
  if (mode_ == 0) {
    slice0 = dynamic_cast<SliceImageTypeUChar* >
      (imgH_.get_rep()->data_.GetPointer());
    volume0 = dynamic_cast<VolumeImageTypeUChar* >
      (volH_.get_rep()->data_.GetPointer());
  } else {
    slice1 = dynamic_cast<SliceImageTypeFloat* >
      (imgH_.get_rep()->data_.GetPointer());
    volume1 = dynamic_cast<VolumeImageTypeFloat* >
      (volH_.get_rep()->data_.GetPointer());
  }


  switch(axis_.get()) {
  case 0:
    {
      if (mode_ == 0) {
	if ((int)(slice0->GetLargestPossibleRegion()).GetSize()[0] != size1_.get() ||
	    (int)(slice0->GetLargestPossibleRegion()).GetSize()[1] != size2_.get()) {
	  error("Size of slice must be same as specified in UI.");
	  return;
	}
      } else {
	if ((int)(slice1->GetLargestPossibleRegion()).GetSize()[0] != size1_.get() ||
	    (int)(slice1->GetLargestPossibleRegion()).GetSize()[1] != size2_.get()) {
	  error("Size of slice must be same as specified in UI.");
	  return;
	}
      }

      if (index_.get() > size0_.get()) {
	error("Index must be within given size dimensions");
	return;
      }
      
      int x = index_.get();	 
      if (mode_ == 0) {
	for(int y=0; y<(int)size1_.get(); y++)
	  for(int z=0; z<(int)size2_.get(); z++) {
	    SliceImageTypeUChar::IndexType slice_pixel;
	    VolumeImageTypeUChar::IndexType vol_pixel;
	    
	    slice_pixel[0] = y;
	    slice_pixel[1] = z;
	    vol_pixel[0] = x;
	    vol_pixel[1] = y;
	    vol_pixel[2] = z;
	    
	    volume0->SetPixel(vol_pixel, slice0->GetPixel(slice_pixel));
	  }
      } else {
      for(int y=0; y<(int)size1_.get(); y++)
	for(int z=0; z<(int)size2_.get(); z++) {
	  SliceImageTypeFloat::IndexType slice_pixel;
	  VolumeImageTypeFloat::IndexType vol_pixel;
	  
	  slice_pixel[0] = y;
	  slice_pixel[1] = z;
	  vol_pixel[0] = x;
	  vol_pixel[1] = y;
	  vol_pixel[2] = z;
	  
	  volume1->SetPixel(vol_pixel, slice1->GetPixel(slice_pixel));
	}
      }
    }
    break;
  case 1:
    {
      if (mode_ == 0) {
	if ((int)(slice0->GetLargestPossibleRegion()).GetSize()[0] != size0_.get() ||
	    (int)(slice0->GetLargestPossibleRegion()).GetSize()[1] != size2_.get()) {
	  error("Size of slice must be same as specified in UI.");
	  return;
	}
      } else {
	if ((int)(slice1->GetLargestPossibleRegion()).GetSize()[0] != size0_.get() ||
	    (int)(slice1->GetLargestPossibleRegion()).GetSize()[1] != size2_.get()) {
	  error("Size of slice must be same as specified in UI.");
	  return;
	}
      }
      if (index_.get() > size1_.get()) {
	error("Index must be within given size dimensions");
	return;
      }
      
      int y = index_.get();
      if (mode_ == 0) {
	for(int x=0; x<(int)size0_.get(); x++)
	  for(int z=0; z<(int)size2_.get(); z++) {
	    SliceImageTypeUChar::IndexType slice_pixel;
	    VolumeImageTypeUChar::IndexType vol_pixel;
	    
	    slice_pixel[0] = x;
	    slice_pixel[1] = z;
	    vol_pixel[0] = x;
	    vol_pixel[1] = y;
	    vol_pixel[2] = z;
	    
	    volume0->SetPixel(vol_pixel, slice0->GetPixel(slice_pixel));
	  }
      } else {
	for(int x=0; x<(int)size0_.get(); x++)
	  for(int z=0; z<(int)size2_.get(); z++) {
	    SliceImageTypeFloat::IndexType slice_pixel;
	    VolumeImageTypeFloat::IndexType vol_pixel;
	    
	    slice_pixel[0] = x;
	    slice_pixel[1] = z;
	    vol_pixel[0] = x;
	    vol_pixel[1] = y;
	    vol_pixel[2] = z;
	    
	    volume1->SetPixel(vol_pixel, slice1->GetPixel(slice_pixel));
	  }
      }
    }
    break;
  case 2:
    {
      if (mode_ == 0) {
	if ((int)(slice0->GetLargestPossibleRegion()).GetSize()[0] != size0_.get() ||
	    (int)(slice0->GetLargestPossibleRegion()).GetSize()[1] != size1_.get()) {
	  error("Size of slice must be same as specified in UI.");
	  return;
	}
      } else {       
	if ((int)(slice1->GetLargestPossibleRegion()).GetSize()[0] != size0_.get() ||
	    (int)(slice1->GetLargestPossibleRegion()).GetSize()[1] != size1_.get()) {
	  error("Size of slice must be same as specified in UI.");
	  return;
	}
      }
      if (index_.get() > size2_.get()) {
	error("Index must be within given size dimensions");
	return;
      }
      
      int z = index_.get();
      if (mode_ == 0) {
	for(int x=0; x<=(int)size0_.get(); x++)
	  for(int y=0; y<=(int)size1_.get(); y++) {
	    
	    SliceImageTypeUChar::IndexType slice_pixel;
	    VolumeImageTypeUChar::IndexType vol_pixel;
	    
	    slice_pixel[0] = x;
	    slice_pixel[1] = y;
	    vol_pixel[0] = x;
	    vol_pixel[1] = y;
	    vol_pixel[2] = z;
	    
	    volume0->SetPixel(vol_pixel, slice0->GetPixel(slice_pixel));
	  }
      } else {
	for(int x=0; x<=(int)size0_.get(); x++)
	  for(int y=0; y<=(int)size1_.get(); y++) {
	    
	    SliceImageTypeFloat::IndexType slice_pixel;
	    VolumeImageTypeFloat::IndexType vol_pixel;
	    
	    slice_pixel[0] = x;
	    slice_pixel[1] = y;
	    vol_pixel[0] = x;
	    vol_pixel[1] = y;
	    vol_pixel[2] = z;
	    
	    volume1->SetPixel(vol_pixel, slice1->GetPixel(slice_pixel));
	  }
      }
    }
    break;
  }
  
  // send output
  ITKDatatype *out = scinew ITKDatatype;
  if (mode_ == 0) {
    out->data_ = volume0;
  } else {
    out->data_ = volume1;
  }
  volH_ = out;
  oimg_->send(volH_);
  
}

void PasteImageFilter::generate_volume()
{
  if (mode_ == 0) {
    typedef itk::Image<unsigned char, 3> ImageType;
    ImageType::Pointer img = ImageType::New();
    
    ImageType::SizeType si;
    ImageType::IndexType in;
    
    si[0] = size0_.get();
    si[1] = size1_.get();
    si[2] = size2_.get();
    
    in.Fill(0);
    
    ImageType::RegionType reg;
    reg.SetSize(si);
    reg.SetIndex(in);
    img->SetRegions(reg);
    img->Allocate();
    img->FillBuffer((unsigned char)fill_value_.get());
    
    ITKDatatype *dt = scinew ITKDatatype;
    dt->data_ = img;
    volH_ = dt;
    generation_ = imgH_->generation;
  } else {
    typedef itk::Image<float, 3> ImageType;
    ImageType::Pointer img = ImageType::New();
    
    ImageType::SizeType si;
    ImageType::IndexType in;
    
    si[0] = size0_.get();
    si[1] = size1_.get();
    si[2] = size2_.get();
    
    in.Fill(0);
    
    ImageType::RegionType reg;
    reg.SetSize(si);
    reg.SetIndex(in);
    img->SetRegions(reg);
    img->Allocate();
    img->FillBuffer((float)fill_value_.get());
    
    ITKDatatype *dt = scinew ITKDatatype;
    dt->data_ = img;
    volH_ = dt;
    generation_ = imgH_->generation;
  }
  old_mode_ = mode_;
}

void PasteImageFilter::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("PasteImageFilter needs a minor command");
    return;
  }

  if( args[1] == "generate_volume" ) 
  {
    generate_volume();
  }
  else 
  {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace Insight


