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
 * C++ (CC) FILE : AnalyzeToNrrd.cc
 *
 * DESCRIPTION   : 
 *                     
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 *         
 * CREATED       : 9/19/2003
 *
 * MODIFIED      : 9/19/2003
 *
 * DOCUMENTATION :
 * 
 * NOTES         : 
 *
 * Copyright (C) 2003 SCI Group
*/

// SCIRun includes

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Teem/share/share.h>
#include <Packages/Teem/Core/Datatypes/NrrdData.h>
#include <Packages/Teem/Dataflow/Ports/NrrdPort.h>

//#ifdef HAVE_INSIGHT
//#include <Core/Algorithms/DataIO/AnalyzeReader.h>
//#include <Core/Algorithms/DataIO/AnalyzeImage.h>
//#endif

namespace SCITeem {

using namespace SCIRun;

// ****************************************************************************
// **************************** Class: AnalyzeToNrrd **************************
// ****************************************************************************

class TeemSHARE AnalyzeToNrrd : public Module {
public:

  //! GUI variables
  GuiString gui_filename_;

  AnalyzeToNrrd(GuiContext*);

  virtual ~AnalyzeToNrrd();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:

  NrrdOPort*      onrrd_;

};


DECLARE_MAKER(AnalyzeToNrrd)

/*===========================================================================*/
// 
// AnalyzeToNrrd
//
// Description : Constructor
//
// Arguments   :
//
// GuiContext* ctx - GUI context
//
AnalyzeToNrrd::AnalyzeToNrrd(GuiContext* ctx)
  : Module("AnalyzeToNrrd", ctx, Source, "DataIO", "Teem"),
    gui_filename_(ctx->subVar("filename"))
{
}

/*===========================================================================*/
// 
// ~AnalyzeToNrrd
//
// Description : Destructor
//
// Arguments   : none
//
AnalyzeToNrrd::~AnalyzeToNrrd(){
}


/*===========================================================================*/
// 
// execute 
//
// Description : The execute function for this module.  This is the control
//               center for the module.  This reads a series of DICOM files,
//               constructs a nrrd with the DICOM data, and sends the nrrd
//               downstream.
//
// Arguments   : none
//
void AnalyzeToNrrd::execute(){

}

/*===========================================================================*/
// 
// tcl_command
//
// Description : The tcl_command function for this module.
//
// Arguments   :
//
// GuiArgs& args - GUI arguments
//
// void* userdata - ???
// 
void AnalyzeToNrrd::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


} // End namespace SCITeem
