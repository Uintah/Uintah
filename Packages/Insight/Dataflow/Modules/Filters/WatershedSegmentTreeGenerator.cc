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
 * WatershedSegmentTreeGenerator.cc
 *
 *   Auto Generated File For itk::watershed::SegmentTreeGenerator
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Insight/share/share.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>

#include <itkWatershedSegmentTreeGenerator.h>

namespace Insight 
{

using namespace SCIRun;

class InsightSHARE WatershedSegmentTreeGenerator : public Module 
{
public:

  // Declare GuiVars
  GuiDouble gui_flood_level_;
  GuiInt gui_merge_;
    
  // Declare Ports
  ITKDatatypeIPort* inport1_;
  ITKDatatypeHandle inhandle1_;

  ITKDatatypeIPort* inport2_;
  ITKDatatypeHandle inhandle2_;
  bool inport2_has_data_;

  ITKDatatypeOPort* outport1_;
  ITKDatatypeHandle outhandle1_;

  
  WatershedSegmentTreeGenerator(GuiContext*);

  virtual ~WatershedSegmentTreeGenerator();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template<class ScalarType > 
  bool run( itk::Object*, itk::Object* );

};


template<class ScalarType>
bool WatershedSegmentTreeGenerator::run( itk::Object *obj1, itk::Object *obj2) 
{
  itk::watershed::SegmentTable<ScalarType> *data1 = dynamic_cast<  itk::watershed::SegmentTable<ScalarType> * >(obj1);
  
  if( !data1 ) {
    return false;
  }
  itk::watershed::EquivalencyTable *data2 = dynamic_cast<  itk::watershed::EquivalencyTable * >(obj2);
  
  if( inport2_has_data_ ) {
    if( !data2 ) {
    return false;
    }
  }
  
  // create a new filter
  typename itk::watershed::SegmentTreeGenerator< ScalarType >::Pointer filter = itk::watershed::SegmentTreeGenerator< ScalarType >::New();

  // set filter 
  
  filter->SetFloodLevel( gui_flood_level_.get() ); 
  
  filter->SetMerge( gui_merge_.get() ); 
     
  // set inputs 

  filter->SetInputSegmentTable( data1 );
   
  filter->SetInputEquivalencyTable( data2 );
   

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
    im->data_ = filter->GetOutputSegmentTree();
    outhandle1_ = im; 
  }
  
  return true;
}


DECLARE_MAKER(WatershedSegmentTreeGenerator)

WatershedSegmentTreeGenerator::WatershedSegmentTreeGenerator(GuiContext* ctx)
  : Module("WatershedSegmentTreeGenerator", ctx, Source, "Filters", "Insight"),
     gui_flood_level_(ctx->subVar("flood_level")),
     gui_merge_(ctx->subVar("merge"))
{
  inport2_has_data_ = false;  
}

WatershedSegmentTreeGenerator::~WatershedSegmentTreeGenerator() 
{
}

void WatershedSegmentTreeGenerator::execute() 
{
  // check input ports
  inport1_ = (ITKDatatypeIPort *)get_iport("Segment_Table");
  if(!inport1_) {
    error("Unable to initialize iport");
    return;
  }

  inport1_->get(inhandle1_);

  if(!inhandle1_.get_rep()) {
    error("No data in inport1_!");			       
    return;
  }

  inport2_ = (ITKDatatypeIPort *)get_iport("Equivalency_Table");
  if(!inport2_) {
    error("Unable to initialize iport");
    return;
  }

  inport2_->get(inhandle2_);

  if(!inhandle2_.get_rep()) {
    remark("No data in optional inport2_!");			       
    inport2_has_data_ = false;
  }
  else {
    inport2_has_data_ = true;
  }

  
  // check output ports
  outport1_ = (ITKDatatypeOPort *)get_oport("Merge_Tree");
  if(!outport1_) {
    error("Unable to initialize oport");
    return;
  }

  // get input
  itk::Object* data1 = inhandle1_.get_rep()->data_.GetPointer();
  itk::Object* data2 = 0;
  if( inport2_has_data_ ) {
    data2 = inhandle2_.get_rep()->data_.GetPointer();
  }
  
  // can we operate on it?
  if(0) { } 
  else if(run< float >( data1, data2 )) { }
  else {
    // error
    error("Incorrect input type");
    return;
  }

  // send the data downstream
  outport1_->send(outhandle1_);
  
}

void WatershedSegmentTreeGenerator::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);

}


} // End of namespace Insight
