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

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkExtractImageFilter.h>

namespace Insight {

using namespace SCIRun;

class ExtractImageFilter : public Module {
public:
  ExtractImageFilter(GuiContext*);

  virtual ~ExtractImageFilter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  void load_gui();

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
  vector<int>     lastmin_;
  vector<int>     lastmax_;
  int             last_generation_;
  ITKDatatypeHandle  last_imgH_;
  ITKDatatypeHandle imgH;
};


DECLARE_MAKER(ExtractImageFilter)
ExtractImageFilter::ExtractImageFilter(GuiContext* ctx)
  : Module("ExtractImageFilter", ctx, Source, "Filters", "Insight"),
  num_dims_(ctx->subVar("num-dims")),
  last_generation_(-1), 
  last_imgH_(0)
{
  // this will get overwritten when tcl side initializes, but 
  // until then make sure it is initialized.
  num_dims_.set(0); 
  load_gui();
}

ExtractImageFilter::~ExtractImageFilter(){
}

void ExtractImageFilter::load_gui() {
  num_dims_.reset();
  if (num_dims_.get() == 0) { return; }

 
  lastmin_.resize(num_dims_.get(), -1);
  lastmax_.resize(num_dims_.get(), -1);  

  if ((int)mins_.size() != num_dims_.get()) {
    for (int a = 0; a < num_dims_.get(); a++) {
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
  }
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
  if (!iimg_->get(imgH))
    return;
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
  InputImageType *n = dynamic_cast<  InputImageType * >(obj_InputImage);
  
  if( !n ) {
    return false;
  }
  
  num_dims_.reset();
  
  if (last_generation_ != imgH->generation) {
    ostringstream str;
    
    load_gui();
    
    bool do_clear = false;
    // if the dim and sizes are the same don't clear.
    if ((num_dims_.get() == (int)n->GetImageDimension())) {
      for (int a = 0; a < num_dims_.get(); a++) {
	int size = (n->GetLargestPossibleRegion()).GetSize()[a];
	if (absmaxs_[a]->get() != (size-1)) {
	  do_clear = true;
	  break;
	}
      }
    } else {
      do_clear = true;
    }
    
    
    if (do_clear) {
      
      lastmin_.clear();
      lastmax_.clear();
      vector<GuiInt*>::iterator iter = mins_.begin();
      while(iter != mins_.end()) {
	delete *iter;
	++iter;
      }
      mins_.clear();
      iter = maxs_.begin();
      while(iter != maxs_.end()) {
	delete *iter;
	++iter;
      }
      maxs_.clear();
      iter = absmaxs_.begin();
      while(iter != absmaxs_.end()) {
	delete *iter;
	++iter;
      }
      absmaxs_.clear();
      gui->execute(id.c_str() + string(" clear_dims"));
      
      
      num_dims_.set((int)n->GetImageDimension());
      num_dims_.reset();
      load_gui();
      gui->execute(id.c_str() + string(" init_dims"));
      
      for (int a = 0; a < num_dims_.get(); a++) {
	maxs_[a]->reset();
      }
      for (int a = 0; a < num_dims_.get(); a++) {
	int size = (n->GetLargestPossibleRegion()).GetSize()[a];
	mins_[a]->set(0);
	absmaxs_[a]->set(size-1);
	maxs_[a]->reset();
	absmaxs_[a]->reset();
      }
      
      str << id.c_str() << " set_max_vals" << endl; 
      gui->execute(str.str());
      
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
  Module::tcl_command(args, userdata);
}

} // End namespace Insight


