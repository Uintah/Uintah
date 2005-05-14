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
 *  ExtractImageFilter.cc:
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

#include <itkExtractImageFilter.h>

namespace Insight {

using namespace SCIRun;

class PSECORESHARE ExtractImageFilter : public Module {
public:
  ExtractImageFilter(GuiContext*);

  virtual ~ExtractImageFilter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template<class InputImageType, class OutputImageType > 
  bool run( itk::Object*   );

private:
  ITKDatatypeIPort*      iimg_;
  ITKDatatypeOPort*      oimg_;
  vector<GuiInt*> mins_;
  vector<GuiInt*> maxs_;
  vector<GuiInt*> absmaxs_;
  GuiInt          num_dims_;
  GuiInt          uis_;
  vector<int>     lastmin_;
  vector<int>     lastmax_;
  int             last_generation_;
  ITKDatatypeHandle  last_imgH_;
  ITKDatatypeHandle imgH;
};


DECLARE_MAKER(ExtractImageFilter)
ExtractImageFilter::ExtractImageFilter(GuiContext* ctx)
  : Module("ExtractImageFilter", ctx, Source, "Filters", "Insight"),
  num_dims_(ctx->subVar("num-dims")), uis_(ctx->subVar("uis")),
  last_generation_(-1), 
  last_imgH_(0)
{
  // this will get overwritten when tcl side initializes, but 
  // until then make sure it is initialized.
  num_dims_.set(0); 
  lastmin_.resize(4,-1);
  lastmax_.resize(4,-1);

  for (int a = 0; a < 4; a++) {
    ostringstream str;
    str << "minDim" << a;
    mins_.push_back(new GuiInt(ctx->subVar(str.str())));
    ostringstream str1;
    str1 << "maxDim" << a;
    maxs_.push_back(new GuiInt(ctx->subVar(str1.str())));
    ostringstream str2;
    str2 << "absmaxDim" << a;
    absmaxs_.push_back(new GuiInt(ctx->subVar(str2.str())));
  }
  //load_gui();
}

ExtractImageFilter::~ExtractImageFilter(){
    mins_.clear();
    maxs_.clear();
    absmaxs_.clear();
    
    lastmin_.clear();
    lastmax_.clear();
}


void ExtractImageFilter::execute(){
  
  update_state(NeedData);
  iimg_ = (ITKDatatypeIPort *)get_iport("InputImage");
  oimg_ = (ITKDatatypeOPort *)get_oport("OutputImage");
  
  if (!iimg_) {
    error("Unable to initialize iport 'InputImage'.");
    return;
  }
  if (!oimg_) {
    error("Unable to initialize oport 'OutputImage'.");
    return;
  }
  if (!iimg_->get(imgH)) {
    error("Empty InputImage");
    return;
  }

  if (!imgH.get_rep()) {
    error("Empty input Image.");
    return;
  }
  
  itk::Object *n = imgH.get_rep()->data_.GetPointer();
  
  // can we operate on it?
  if (0) {}
  else if(run<itk::Image<float,2>, itk::Image<float,2> >( n)) {}
  else if(run<itk::Image<float,3>, itk::Image<float,3> >( n)) {}
  else if(run<itk::Image<double,2>, itk::Image<double,2> >( n)) {}
  else if(run<itk::Image<double,3>, itk::Image<double,3> >( n)) {}
  else if(run<itk::Image<unsigned char,2>, itk::Image<unsigned char,2> >( n)) {}
  else if(run<itk::Image<unsigned char,3>, itk::Image<unsigned char,3> >( n)) {}
  else if(run<itk::Image<char,2>, itk::Image<char,2> >( n)) {}
  else if(run<itk::Image<char,3>, itk::Image<char,3> >( n)) {}
  else if(run<itk::Image<unsigned short,2>, itk::Image<unsigned short,2> >( n)) {}
  else if(run<itk::Image<unsigned short,3>, itk::Image<unsigned short,3> >( n)) {}
  else if(run<itk::Image<short,2>, itk::Image<short,2> >( n)) {}
  else if(run<itk::Image<short,3>, itk::Image<short,3> >( n)) {}
  else if(run<itk::Image<int,2>, itk::Image<int,2> >( n)) {}
  else if(run<itk::Image<int,3>, itk::Image<int,3> >( n)) {}
  else if(run<itk::Image<unsigned long,2>, itk::Image<unsigned long,2> >( n)) {}
  else if(run<itk::Image<unsigned long,3>, itk::Image<unsigned long,3> >( n)) {}
  else {
    // error
    error("Incorect input type");
    return;
  }
}


template<class InputImageType, class OutputImageType>
bool ExtractImageFilter::run( itk::Object *obj_InputImage) 
{
  InputImageType *img = dynamic_cast<  InputImageType * >(obj_InputImage);
  if( !img ) {
    return false;
  }

  // Copy the input image so that upstream modules aren't affected
  typename InputImageType::Pointer n = InputImageType::New();
  typename InputImageType::SizeType si;
  typename InputImageType::IndexType in;

  for(int a=0; a<(int)img->GetImageDimension(); a++) {
    si[a] = (img->GetRequestedRegion()).GetSize()[a];
    in[a] = 0;
  }

  typename InputImageType::RegionType reg;
  reg.SetSize(si);
  reg.SetIndex(in);
  n->SetRegions(reg);
  n->Allocate();

  if ((int)img->GetImageDimension() == 3) {
    typename InputImageType::IndexType pixel;
    for(int x=0; x<(int)si[0];x++)
      for(int y=0; y<(int)si[1];y++)
	for(int z=0; z<(int)si[2];z++) {
	  pixel[0] = x;
	  pixel[1] = y;
	  pixel[2] = z;
	  typename InputImageType::PixelType val = img->GetPixel(pixel);
	  n->SetPixel(pixel,val);
	}
  } else if ((int)img->GetImageDimension() == 2) {
    typename InputImageType::IndexType pixel;
    for(int x=0; x<(int)si[0];x++)
      for(int y=0; y<(int)si[1];y++) {
	pixel[0] = x;
	pixel[1] = y;
	typename InputImageType::PixelType val = img->GetPixel(pixel);
	n->SetPixel(pixel,val);
      }  
  } else {
    error("ExtractImageFilter only works on 2 and 3D data\n");
    return false;
  }

  num_dims_.reset();

  bool new_dataset = (last_generation_ != imgH->generation);
  bool first_time = (last_generation_ == -1);

  // create any axes that might have been saved
  if (first_time) {
    uis_.reset();
    for(int i=4; i<uis_.get(); i++) {
      ostringstream str, str2, str3, str4;
      str << "minDim" << i;
      str2 << "maxDim" << i;
      str3 << "absmaxDim" << i;
      str4 << i;
      mins_.push_back(new GuiInt(ctx->subVar(str.str())));
      maxs_.push_back(new GuiInt(ctx->subVar(str2.str())));
      absmaxs_.push_back(new GuiInt(ctx->subVar(str3.str())));

      mins_[i]->reset();
      maxs_[i]->reset();
      lastmin_.push_back(mins_[i]->get());
      lastmax_.push_back(maxs_[i]->get());  

      gui->execute(id.c_str() + string(" make_min_max " + str4.str()));
    }
  }

  last_generation_ = imgH->generation;
  num_dims_.set((int)n->GetImageDimension());
  num_dims_.reset();

  // remove any unused uis or add any needes uis
  if (uis_.get() > (int)n->GetImageDimension()) {
    // remove them
    for(int i=uis_.get()-1; i>=(int)n->GetImageDimension(); i--) {
      ostringstream str;
      str << i;
      vector<GuiInt*>::iterator iter = mins_.end();
      vector<GuiInt*>::iterator iter2 = maxs_.end();
      vector<GuiInt*>::iterator iter3 = absmaxs_.end();
      vector<int>::iterator iter4 = lastmin_.end();
      vector<int>::iterator iter5 = lastmax_.end();
      mins_.erase(iter, iter);
      maxs_.erase(iter2, iter2);
      absmaxs_.erase(iter3, iter3);

      lastmin_.erase(iter4, iter4);
      lastmax_.erase(iter5, iter5);

      gui->execute(id.c_str() + string(" clear_axis " + str.str()));
    }
    uis_.set((int)n->GetImageDimension());
  } else if (uis_.get() < (int)n->GetImageDimension()) {
    for (int i=uis_.get(); i < num_dims_.get(); i++) {
      ostringstream str, str2, str3, str4;
      str << "minDim" << i;
      str2 << "maxDim" << i;
      str3 << "absmaxDim" << i;
      str4 << i;
      mins_.push_back(new GuiInt(ctx->subVar(str.str())));
      maxs_.push_back(new GuiInt(ctx->subVar(str2.str())));
      maxs_[i]->set((n->GetLargestPossibleRegion()).GetSize()[i]-1);
      absmaxs_.push_back(new GuiInt(ctx->subVar(str3.str())));
      absmaxs_[i]->set((n->GetLargestPossibleRegion()).GetSize()[i]-1);
      
      lastmin_.push_back(0);
      lastmax_.push_back((n->GetLargestPossibleRegion()).GetSize()[i]-1); 
      
      gui->execute(id.c_str() + string(" make_min_max " + str4.str()));
    }
    uis_.set((int)n->GetImageDimension());
  }
 
  if (new_dataset) {
    for (int a=0; a<num_dims_.get(); a++) {
      int max = (n->GetLargestPossibleRegion()).GetSize()[a];
      maxs_[a]->reset();
      absmaxs_[a]->set(n->GetLargestPossibleRegion().GetSize()[a]-1);
      absmaxs_[a]->reset();
      if (maxs_[a]->get() > max) {
	warning("Out of bounds, resetting axis min/max");
	mins_[a]->set(0);
	mins_[a]->reset();
	maxs_[a]->set(max);
	maxs_[a]->reset();
	lastmin_[a] = 0;
	lastmax_[a] = max;
      }
    }

    gui->execute(id.c_str() + string (" update_sizes "));    
  }

//   if (new_dataset && !first_time) {
//     ostringstream str;
//     str << id.c_str() << " reset_vals" << endl; 
//     gui->execute(str.str());  
//   }

  for (int a=0; a<num_dims_.get(); a++) {
    mins_[a]->reset();
    maxs_[a]->reset();
    absmaxs_[a]->reset();
  }


  // See if any of the sizes have changed.
  bool update = new_dataset;

  for (int i=0; i<num_dims_.get(); i++) {
      mins_[i]->reset();

    int min = mins_[i]->get();
    if (lastmin_[i] != min) {
      update = true;
      lastmin_[i] = min;
    }
    maxs_[i]->reset();
    int max = maxs_[i]->get();
    if (lastmax_[i] != max) {
      update = true;
    }
  }

  if (num_dims_.get() == 0) { return true; }
  
  for (int a = 0; a < num_dims_.get(); a++) {
    mins_[a]->reset();
    maxs_[a]->reset();
    absmaxs_[a]->reset();
  }
  
  
  
  if (last_generation_ == imgH->generation && last_imgH_.get_rep()) {
    bool same = true;
    for (int i = 0; i < num_dims_.get(); i++) {
      if (lastmin_[i] != mins_[i]->get()) {
	same = false;
	lastmin_[i] = mins_[i]->get();
      }
      if (lastmax_[i] != maxs_[i]->get()) {
	same = false;
	lastmax_[i] = maxs_[i]->get();
      }
    }
    if (same) {
      oimg_->send(last_imgH_);
      return true;
    }
  }
  last_generation_ = imgH->generation;
  
  
  typedef itk::ExtractImageFilter<InputImageType, OutputImageType> FilterType;
  
  typename FilterType::Pointer filter = FilterType::New();

  filter->SetInput(n);
  
  // define a region
  typename InputImageType::IndexType start;
  typename InputImageType::IndexType input_start;
  
  typename InputImageType::SizeType size;
  for(int i=0; i<(int)n->GetImageDimension(); i++) {
    start[i] = mins_[i]->get();
    size[i] = maxs_[i]->get()-start[i];
    
    input_start[i] = (n->GetLargestPossibleRegion()).GetIndex()[i];
  }

  typename InputImageType::RegionType region;
  region.SetSize( size );
  region.SetIndex( start );
  
  filter->SetExtractionRegion( region );
  
  // execute the filter
  try {
    
    filter->Update();
    
  } catch ( itk::ExceptionObject & err ) {
    error("ExceptionObject caught!");
    error(err.GetDescription());
  }
  
  // set new start index to what input image was
  //region.SetIndex( input_start );
  //filter->GetOutput()->SetRegions( region );
  
  ITKDatatype *nrrd = scinew ITKDatatype;
  nrrd->data_ = filter->GetOutput();
  last_imgH_ = nrrd;
  oimg_->send(last_imgH_);
  return true;
}

void ExtractImageFilter::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("ExtractImageFilter needs a minor command");
    return;
  }

  if( args[1] == "add_axis" ) 
  {
      uis_.reset();
      int i = uis_.get();
      ostringstream str, str2, str3, str4;
      str << "minDim" << i;
      str2 << "maxDim" << i;
      str3 << "absmaxDim" << i;
      str4 << i;
      mins_.push_back(new GuiInt(ctx->subVar(str.str())));
      maxs_.push_back(new GuiInt(ctx->subVar(str2.str())));
      absmaxs_.push_back(new GuiInt(ctx->subVar(str3.str())));

      lastmin_.push_back(0);
      lastmax_.push_back(0); 

      gui->execute(id.c_str() + string(" make_min_max " + str4.str()));

      uis_.set(uis_.get() + 1);
  }
  else if( args[1] == "remove_axis" ) 
  {
    uis_.reset();
    int i = uis_.get()-1;
    ostringstream str;
    str << i;
    vector<GuiInt*>::iterator iter = mins_.end();
    vector<GuiInt*>::iterator iter2 = maxs_.end();
    vector<GuiInt*>::iterator iter3 = absmaxs_.end();
    vector<int>::iterator iter4 = lastmin_.end();
    vector<int>::iterator iter5 = lastmax_.end();
    mins_.erase(iter, iter);
    maxs_.erase(iter2, iter2);
    absmaxs_.erase(iter3, iter3);
    
    lastmin_.erase(iter4, iter4);
    lastmax_.erase(iter5, iter5);
    
    gui->execute(id.c_str() + string(" clear_axis " + str.str()));
    uis_.set(uis_.get() - 1);
  }
  else 
  {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace Insight


