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
 *  Gradient.cc:  Unfinished modules
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
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class Gradient : public Module
{
private:
  FieldIPort *ifp;
  FieldOPort *ofp;
  GuiInt  interpolate_;

public:
  Gradient(const string& id);
  virtual ~Gradient();

  virtual void execute();

  template <class F> void dispatch_tetvol(F *f);
  template <class F> void dispatch_latticevol(F *f);
};


extern "C" Module* make_Gradient(const string& id) {
  return new Gradient(id);
}


Gradient::Gradient(const string& id)
  : Module("Gradient", id, Filter, "Fields", "SCIRun"),
    interpolate_("interpolate", id, this)
{
}



Gradient::~Gradient()
{
}

template <class F>
void
Gradient::dispatch_tetvol(F *f)
{
  TetVolMeshHandle tvm = f->get_typed_mesh(); 
  TetVol<Vector> *result = new TetVol<Vector>(tvm, Field::CELL);
  typename F::mesh_type::Cell::iterator ci, cie;
  tvm->begin(ci); tvm->end(cie);
  while (ci != cie)
  {
    result->set_value(f->cell_gradient(*ci), *ci);
    ++ci;
  }

  result->freeze();
  FieldHandle fh(result);
  ofp->send(fh);
}

template <class F>
void
Gradient::dispatch_latticevol(F *f)
{
  LatVolMeshHandle lvm = f->get_typed_mesh(); 
  LatticeVol<Vector> *result = new LatticeVol<Vector>(lvm, Field::CELL);
  typename F::mesh_type::Cell::iterator ci, cie;
  lvm->begin(ci); lvm->end(cie);
  while (ci != cie)
  {
    Point p;
    lvm->get_center(p, *ci);
    Vector v;
    f->get_gradient(v, p);
    result->set_value(v, *ci);
    ++ci;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Gradient");
  if (!ofp) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  result->freeze();
  FieldHandle fh(result);
  ofp->send(fh);
}


void
Gradient::execute()
{
  ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle fieldhandle;
  Field *field;
  if (!ifp) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!(ifp->get(fieldhandle) && (field = fieldhandle.get_rep())))
  {
    return;
  }

  ofp = (FieldOPort *)get_oport("Output Gradient");
  if (!ofp) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  const TypeDescription *ftd = fieldhandle->get_type_description();

  if (ftd->get_name() == get_type_description((TetVol<double> *)0)->get_name())
  {
    dispatch_tetvol((TetVol<double> *)field);
  }
  else if (ftd->get_name() == get_type_description((TetVol<int> *)0)->get_name())
  {
    dispatch_tetvol((TetVol<int> *)field);
  }
  else if (ftd->get_name() == get_type_description((TetVol<short> *)0)->get_name())
  {
    dispatch_tetvol((TetVol<short> *)field);
  }
  else if (ftd->get_name() == get_type_description((TetVol<unsigned char> *)0)->get_name())
  {
    dispatch_tetvol((TetVol<unsigned char> *)field);
  }
  else if (ftd->get_name() == get_type_description((LatticeVol<double> *)0)->get_name())
  {
    dispatch_latticevol((LatticeVol<double> *)field);
  }
  else if (ftd->get_name() == get_type_description((LatticeVol<int> *)0)->get_name())
  {
    dispatch_latticevol((LatticeVol<int> *)field);
  }
  else if (ftd->get_name() == get_type_description((LatticeVol<short> *)0)->get_name())
  {
    dispatch_latticevol((LatticeVol<short> *)field);
  }
  else if (ftd->get_name() == get_type_description((LatticeVol<unsigned char> *)0)->get_name())
  {
    dispatch_latticevol((LatticeVol<unsigned char> *)field);
  }
  else
  {
    error("Unable to handle a field of type '" + ftd->get_name() + "'.");
  }
}


} // End namespace SCIRun

