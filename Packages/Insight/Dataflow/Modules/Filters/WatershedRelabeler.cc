/*
 * WatershedRelabeler.cc
 *
 *   Auto Generated File For itk::watershed::Relabeler
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Insight/share/share.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkWatershedRelabeler.h>

namespace Insight 
{

using namespace SCIRun;

class InsightSHARE WatershedRelabeler : public Module 
{
public:

  // Declare GuiVars
  GuiDouble gui_flood_level_;
    
  // Declare Ports
  ITKDatatypeIPort* inport1_;
  ITKDatatypeHandle inhandle1_;

  ITKDatatypeIPort* inport2_;
  ITKDatatypeHandle inhandle2_;

  ITKDatatypeOPort* outport1_;
  ITKDatatypeHandle outhandle1_;

  
  WatershedRelabeler(GuiContext*);

  virtual ~WatershedRelabeler();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template<class ScalarType, unsigned int ImageDimension > 
  bool run( itk::Object*, itk::Object* );

};


template<class ScalarType, unsigned int ImageDimension>
bool WatershedRelabeler::run( itk::Object *obj1, itk::Object *obj2) 
{
  itk::Image<unsigned long, ImageDimension> *data1 = dynamic_cast<  itk::Image<unsigned long, ImageDimension> * >(obj1);
  
  if( !data1 ) {
    return false;
  }
  itk::watershed::SegmentTree<ScalarType> *data2 = dynamic_cast<  itk::watershed::SegmentTree<ScalarType> * >(obj2);
  
  if( !data2 ) {
    return false;
  }

  // create a new filter
  typename itk::watershed::Relabeler< ScalarType, ImageDimension >::Pointer filter = itk::watershed::Relabeler< ScalarType, ImageDimension >::New();

  // set filter 
  
  filter->SetFloodLevel( gui_flood_level_.get() ); 
     
  // set inputs 

  filter->SetInputImage( data1 );
   
  filter->SetInputSegmentTree( data2 );
   

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
  
  return true;
}


DECLARE_MAKER(WatershedRelabeler)

WatershedRelabeler::WatershedRelabeler(GuiContext* ctx)
  : Module("WatershedRelabeler", ctx, Source, "Filters", "Insight"),
     gui_flood_level_(ctx->subVar("flood_level"))
{

}

WatershedRelabeler::~WatershedRelabeler() 
{
}

void WatershedRelabeler::execute() 
{
  // check input ports
  inport1_ = (ITKDatatypeIPort *)get_iport("Labeled_Image");
  if(!inport1_) {
    error("Unable to initialize iport");
    return;
  }

  inport1_->get(inhandle1_);

  if(!inhandle1_.get_rep()) {
    error("No data in inport1_!");			       
    return;
  }

  inport2_ = (ITKDatatypeIPort *)get_iport("Segment_Tree");
  if(!inport2_) {
    error("Unable to initialize iport");
    return;
  }

  inport2_->get(inhandle2_);

  if(!inhandle2_.get_rep()) {
    error("No data in inport2_!");			       
    return;
  }


  // check output ports
  outport1_ = (ITKDatatypeOPort *)get_oport("Relabeled_Image");
  if(!outport1_) {
    error("Unable to initialize oport");
    return;
  }

  // get input
  itk::Object* data1 = inhandle1_.get_rep()->data_.GetPointer();
  itk::Object* data2 = inhandle2_.get_rep()->data_.GetPointer();
  
  // can we operate on it?
  if(0) { } 
  else if(run< float, 3 >( data1, data2 )) { }
  else {
    // error
    error("Incorrect input type");
    return;
  }

  // send the data downstream
  outport1_->send(outhandle1_);
  
}

void WatershedRelabeler::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);

}


} // End of namespace Insight
