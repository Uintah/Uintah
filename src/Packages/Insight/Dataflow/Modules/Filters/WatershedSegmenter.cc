/*
 * WatershedSegmenter.cc
 *
 *   Auto Generated File For itk::watershed::Segmenter
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Insight/share/share.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkWatershedSegmenter.h>

namespace Insight 
{

using namespace SCIRun;

class InsightSHARE WatershedSegmenter : public Module 
{
public:

  // Declare GuiVars
  GuiDouble gui_threshold_;
    
  // Declare Ports
  ITKDatatypeIPort* inport1_;
  ITKDatatypeHandle inhandle1_;

  ITKDatatypeOPort* outport1_;
  ITKDatatypeHandle outhandle1_;

  ITKDatatypeOPort* outport2_;
  ITKDatatypeHandle outhandle2_;

  ITKDatatypeOPort* outport3_;
  ITKDatatypeHandle outhandle3_;

  
  WatershedSegmenter(GuiContext*);

  virtual ~WatershedSegmenter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template<class InputImageType > 
  bool run( itk::Object* );

};


template<class InputImageType>
bool WatershedSegmenter::run( itk::Object *obj1) 
{
  InputImageType *data1 = dynamic_cast<  InputImageType * >(obj1);
  
  if( !data1 ) {
    return false;
  }

  // create a new filter
  typename itk::watershed::Segmenter< InputImageType >::Pointer filter = itk::watershed::Segmenter< InputImageType >::New();

  // set filter 
  
  filter->SetThreshold( gui_threshold_.get() ); 
     
  // set inputs 

  filter->SetInputImage( data1 );
   

  // execute the filter
  try {

    filter->Update();

  } catch ( itk::ExceptionObject & err ) {
     error("ExceptionObject caught!");
     error(err.GetDescription());
  }

  // get filter output
  
  if(!outhandle1_.get_rep())
  {
    ITKDatatype* im = scinew ITKDatatype;
    im->data_ = filter->GetOutputImage();
    outhandle1_ = im; 
  }
  
  if(!outhandle2_.get_rep())
  {
    ITKDatatype* im = scinew ITKDatatype;
    im->data_ = filter->GetSegmentTable();
    outhandle2_ = im; 
  }
  
  if(!outhandle3_.get_rep())
  {
    ITKDatatype* im = scinew ITKDatatype;
    im->data_ = filter->GetBoundary();
    outhandle3_ = im; 
  }
  
  return true;
}


DECLARE_MAKER(WatershedSegmenter)

WatershedSegmenter::WatershedSegmenter(GuiContext* ctx)
  : Module("WatershedSegmenter", ctx, Source, "Filters", "Insight"),
     gui_threshold_(ctx->subVar("threshold"))
{

}

WatershedSegmenter::~WatershedSegmenter() 
{
}

void WatershedSegmenter::execute() 
{
  // check input ports
  inport1_ = (ITKDatatypeIPort *)get_iport("Scalar_Image");
  if(!inport1_) {
    error("Unable to initialize iport");
    return;
  }

  inport1_->get(inhandle1_);

  if(!inhandle1_.get_rep()) {
    error("No data in inport1_!");			       
    return;
  }


  // check output ports
  outport1_ = (ITKDatatypeOPort *)get_oport("Labeled_Image");
  if(!outport1_) {
    error("Unable to initialize oport");
    return;
  }
  outport2_ = (ITKDatatypeOPort *)get_oport("Segment_Table");
  if(!outport2_) {
    error("Unable to initialize oport");
    return;
  }
  outport3_ = (ITKDatatypeOPort *)get_oport("Boundary");
  if(!outport3_) {
    error("Unable to initialize oport");
    return;
  }

  // get input
  itk::Object* data1 = inhandle1_.get_rep()->data_.GetPointer();
  
  // can we operate on it?
  if(0) { } 
  else if(run< itk::Image<float, 3> >( data1 )) { }
  else {
    // error
    error("Incorrect input type");
    return;
  }

  // send the data downstream
  outport1_->send(outhandle1_);
  outport2_->send(outhandle2_);
  outport3_->send(outhandle3_);
  
}

void WatershedSegmenter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);

}


} // End of namespace Insight
