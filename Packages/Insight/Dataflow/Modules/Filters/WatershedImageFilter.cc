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
 * WatershedImageFilter.cc
 *
 *   Auto Generated File For itk::WatershedImageFilter
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Insight/share/share.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkWatershedImageFilter.h>

namespace Insight 
{

using namespace SCIRun;

class InsightSHARE WatershedImageFilter : public Module 
{
public:

  // Declare GuiVars
  GuiDouble gui_threshold_;
  GuiDouble gui_level_;
    
  // Declare Ports
  ITKDatatypeIPort* inport1_;
  ITKDatatypeHandle inhandle1_;

  ITKDatatypeOPort* outport1_;
  ITKDatatypeHandle outhandle1_;

  
  WatershedImageFilter(GuiContext*);

  virtual ~WatershedImageFilter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template<class InputImageType > 
  bool run( itk::Object* );

};


template<class InputImageType>
bool WatershedImageFilter::run( itk::Object *obj1) 
{
  InputImageType *data1 = dynamic_cast<  InputImageType * >(obj1);
  
  if( !data1 ) {
    return false;
  }

  // create a new filter
  typename itk::WatershedImageFilter< InputImageType >::Pointer filter = itk::WatershedImageFilter< InputImageType >::New();

  // set filter 
  
  filter->SetThreshold( gui_threshold_.get() ); 
  
  filter->SetLevel( gui_level_.get() ); 
     
  // set inputs 

  filter->SetInput( data1 );
   

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
    im->data_ = filter->GetBasicSegmentation();
    outhandle1_ = im; 
  }
  
  return true;
}


DECLARE_MAKER(WatershedImageFilter)

WatershedImageFilter::WatershedImageFilter(GuiContext* ctx)
  : Module("WatershedImageFilter", ctx, Source, "Filters", "Insight"),
     gui_threshold_(ctx->subVar("threshold")),
     gui_level_(ctx->subVar("level"))
{

}

WatershedImageFilter::~WatershedImageFilter() 
{
}

void WatershedImageFilter::execute() 
{
  // check input ports
  inport1_ = (ITKDatatypeIPort *)get_iport("InputImage");
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
  outport1_ = (ITKDatatypeOPort *)get_oport("OutputImage");
  if(!outport1_) {
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
  
}

void WatershedImageFilter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);

}


} // End of namespace Insight
