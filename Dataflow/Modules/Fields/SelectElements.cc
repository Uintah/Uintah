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

//    File   : SelectElements.cc
//    Author : David Weinstein
//    Date   : August 2001

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Modules/Fields/SelectElements.h>
#include <math.h>

#include <Core/share/share.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class PSECORESHARE SelectElements : public Module {
public:
  GuiString value_;
  GuiInt keep_all_nodes_;
  SelectElements(GuiContext* ctx);
  virtual ~SelectElements();
  virtual void execute();
};

  DECLARE_MAKER(SelectElements)

SelectElements::SelectElements(GuiContext* ctx)
  : Module("SelectElements", ctx, Source, "Fields", "SCIRun"),
    value_(ctx->subVar("value")), 
    keep_all_nodes_(ctx->subVar("keep-all-nodes"))
{
}

SelectElements::~SelectElements(){
}

void SelectElements::execute()
{
  // must find ports and have valid data on inputs
  FieldIPort *ifieldPort = (FieldIPort*)get_iport("Input Field");

  if (!ifieldPort) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  FieldHandle field;
  if (!ifieldPort->get(field) || !field.get_rep()) return;

  ScalarFieldInterface *sfi = field->query_scalar_interface(this);
  if (sfi == 0)
  {
    error("Only works on scalar fields.");
    return;
  }
  if (field->data_at() != Field::CELL) {
    error("Only works for data_at == Field::CELL");
    return;
  }
  double min, max;
  sfi->compute_min_max(min, max);
  string value = value_.get();
  char **v;
  char *value_str = new char[30];
  v = &value_str;
  strcpy(value_str, value.c_str());
  char *matl;
  Array1<int> values;
  while ((matl = strtok(value_str, " ,"))) {
    value_str=0;
    values.add(atoi(matl));
  }
  delete[] (*v);

  int ii;
  for (ii=0; ii<values.size(); ii++) {
    if (values[ii] < min) {
      msgStream_ << "Error - min="<<min<<" value="<<values[ii]<<"\n";
      values[ii]=(int)min;
    } else if (values[ii] > max) {
      msgStream_ << "Error - max="<<max<<" value="<<values[ii]<<"\n";
      values[ii]=(int)max;
    }
  }

  FieldOPort *ofieldPort = (FieldOPort*)get_oport("Output Field");
  if (!ofieldPort) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }

  MatrixOPort *omat1Port = (MatrixOPort*)get_oport("LeadFieldRestrictColumns");
  if (!omat1Port) {
    error("Unable to initialize oport 'LeadFieldRestrictColumns'.");
    return;
  }

  MatrixOPort *omat2Port = (MatrixOPort*)get_oport("LeadFieldInflateRows");
  if (!omat2Port) {
    error("Unable to initialize oport 'LeadFieldInflateRows'.");
    return;
  }

  const TypeDescription *fsrc_td = field->get_type_description();
  const TypeDescription *msrc_td = field->mesh()->get_type_description();
  CompileInfoHandle ci = SelectElementsAlgo::get_compile_info(fsrc_td,msrc_td);
  Handle<SelectElementsAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  Array1<int> elem_valid;
  Array1<int> indices;
  int count;
  int keep_all_nodes=keep_all_nodes_.get();

  FieldHandle tvH(algo->execute(field, elem_valid, indices, count, 
				values, keep_all_nodes));
  ofieldPort->send(tvH);

  ColumnMatrix *cm = scinew ColumnMatrix(indices.size()*3);
  for (ii=0; ii<indices.size(); ii++) {
    (*cm)[ii*3]=indices[ii]*3;
    (*cm)[ii*3+1]=indices[ii]*3+1;
    (*cm)[ii*3+2]=indices[ii]*3+2;
  }
  MatrixHandle cmH(cm);
  omat1Port->send(cmH);

  int k=0;
  ColumnMatrix *cm2 = scinew ColumnMatrix(count*3);
  for (ii=0; ii<count; ii++) {
    if (elem_valid[ii]) {
      (*cm2)[ii*3]=k;
      (*cm2)[ii*3+1]=k+1;
      (*cm2)[ii*3+2]=k+2;
      k+=3;
    } else {
      (*cm2)[ii*3]=-1;
      (*cm2)[ii*3+1]=-1;
      (*cm2)[ii*3+2]=-1;
    }
  }
  MatrixHandle cmH2(cm2);
  omat2Port->send(cmH2);
}    

CompileInfoHandle
SelectElementsAlgo::get_compile_info(const TypeDescription *field_td,
				     const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class("SelectElementsAlgoT");
  static const string base_class_name("SelectElementsAlgo");
  
  CompileInfo *rval = 
    scinew CompileInfo(template_class + "." +
		       field_td->get_filename() + "." +
		       mesh_td->get_filename() + ".",
		       base_class_name, 
		       template_class,
                       field_td->get_name() + ", " + mesh_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
  return rval;
}
} // end SCIRun
