/*
 *  ManageFieldData: Store/retrieve values from an input matrix to/from 
 *            the data of a field
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Core/Containers/String.h>
#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>

#include <iostream>
using std::cerr;
#include <stdio.h>

namespace SCIRun {

class ManageFieldData : public Module
{
  GuiString     data_op_gui_;
  GuiString     data_loc_gui_;
  GuiString     data_type_gui_;

public:
  ManageFieldData(const clString& id);
  virtual ~ManageFieldData();

  virtual void execute();
};


extern "C" Module* make_ManageFieldData(const clString& id)
{
  return new ManageFieldData(id);
}

ManageFieldData::ManageFieldData(const clString& id)
  : Module("ManageFieldData", id, Filter, "Fields", "SCIRun"),
    data_op_gui_("data_op_gui", id, this),
    data_loc_gui_("data_loc_gui", id, this),
    data_type_gui_("data_type_gui", id, this)
{
}



ManageFieldData::~ManageFieldData()
{
}



void
ManageFieldData::execute()
{
#if 0
  update_state(NeedData);

  FieldHandle fieldH;
  if (!ifield_->get(fieldH)) {
    return;
  }

  clString data_op_gui = data_type_gui_.get();

  if (data_op_gui == "getdata") {
    MatrixHandle matrixH;
    if (!imatrix_->get(matrixH)) {
      return;
    }

    update_state(JustStarted);

    MatrixHandle omatrixH;

    // FIXME: pack up all of the data from fieldH->FData into omatrixH
    //    and send it out omatrix_

    // ...

    omatrix_->send(omatrixH);
  } else {   // field_op_gui == "setdata"
    clString data_loc_gui = data_loc_gui_.get();
    clString data_type_gui = data_type_gui_.get();
  
    update_state(JustStarted);

    //    fieldH.detach();    // just in case someone else is sharing this handle

    // FIXME: based on the mapping of matrix values to field data,
    //   as indicated in the field_*_gui variables, set fieldH->FData

    // ...

    ofield_->send(fieldH);
  }
#endif
}

} // End namespace SCIRun
