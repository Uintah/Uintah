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
 *  VectorMagnitude.cc:  Unfinished modules
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/TetVolField.h>
#include <iostream>

namespace SCIRun {

class VectorMagnitude : public Module
{
private:
  FieldIPort *ifp;
  FieldOPort *ofp;

public:
  VectorMagnitude(const string& id);
  virtual ~VectorMagnitude();

  virtual void execute();
};


extern "C" Module* make_VectorMagnitude(const string& id) {
  return new VectorMagnitude(id);
}


VectorMagnitude::VectorMagnitude(const string& id)
  : Module("VectorMagnitude", id, Filter, "Fields", "SCIRun")
{
}

VectorMagnitude::~VectorMagnitude()
{
}

void
VectorMagnitude::execute()
{
  ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle fieldin;
  Field *field;
  if (!ifp) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!(ifp->get(fieldin) && (field = fieldin.get_rep())))
  {
    return;
  }

  ofp = (FieldOPort *)get_oport("Output VectorMagnitude");
  if (!ofp) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  FieldHandle fieldout;
  TetVolField<Vector> *tvv = dynamic_cast<TetVolField<Vector> *>(field);
  if (!tvv) {
    cerr << "VectorMagnitude only works on TetVolField<Vector> for now...\n";
    return;
  }
  TetVolField<double> *tvd = 
    scinew TetVolField<double>(tvv->get_typed_mesh(), tvv->data_at());
  TetVolField<Vector>::fdata_type::iterator in = tvv->fdata().begin();
  TetVolField<double>::fdata_type::iterator out = tvd->fdata().begin();
  TetVolField<Vector>::fdata_type::iterator end = tvv->fdata().end();
  while (in != end) {
    double l=in->length();
    *out = l;
    ++in; ++out;
  }
  ofp->send(tvd);
}
} // End namespace SCIRun

