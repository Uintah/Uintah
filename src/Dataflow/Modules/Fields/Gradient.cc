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
#include <Core/Datatypes/QuadraticTetVolField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/LatVolField.h>
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

  template <class F> void dispatch_quadratictetvol(F *f);
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
Gradient::dispatch_quadratictetvol(F *f)
{
  QuadraticTetVolMeshHandle tvm = f->get_typed_mesh(); 
  QuadraticTetVolField<Vector> *result = new QuadraticTetVolField<Vector>(tvm, Field::CELL);
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
Gradient::dispatch_tetvol(F *f)
{
  TetVolMeshHandle tvm = f->get_typed_mesh(); 
  TetVolField<Vector> *result = new TetVolField<Vector>(tvm, Field::CELL);
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
  LatVolField<Vector> *result = new LatVolField<Vector>(lvm, Field::CELL);
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
  if (ftd->get_name() == get_type_description((QuadraticTetVolField<double> *)0)->get_name())
  {
    dispatch_quadratictetvol((QuadraticTetVolField<double> *)field);
  }
  else if (ftd->get_name() == get_type_description((TetVolField<double> *)0)->get_name())
  {
    dispatch_tetvol((TetVolField<double> *)field);
  }
  else if (ftd->get_name() == get_type_description((TetVolField<int> *)0)->get_name())
  {
    dispatch_tetvol((TetVolField<int> *)field);
  }
  else if (ftd->get_name() == get_type_description((TetVolField<short> *)0)->get_name())
  {
    dispatch_tetvol((TetVolField<short> *)field);
  }
  else if (ftd->get_name() == get_type_description((TetVolField<unsigned char> *)0)->get_name())
  {
    dispatch_tetvol((TetVolField<unsigned char> *)field);
  }
  else if (ftd->get_name() == get_type_description((LatVolField<double> *)0)->get_name())
  {
    dispatch_latticevol((LatVolField<double> *)field);
  }
  else if (ftd->get_name() == get_type_description((LatVolField<int> *)0)->get_name())
  {
    dispatch_latticevol((LatVolField<int> *)field);
  }
  else if (ftd->get_name() == get_type_description((LatVolField<short> *)0)->get_name())
  {
    dispatch_latticevol((LatVolField<short> *)field);
  }
  else if (ftd->get_name() == get_type_description((LatVolField<unsigned char> *)0)->get_name())
  {
    dispatch_latticevol((LatVolField<unsigned char> *)field);
  }
  else
  {
    error("Unable to handle a field of type '" + ftd->get_name() + "'.");
  }
}


} // End namespace SCIRun

