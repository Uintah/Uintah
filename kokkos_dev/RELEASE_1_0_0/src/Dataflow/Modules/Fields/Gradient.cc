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

using std::cerr;

namespace SCIRun {

class Gradient : public Module
{
private:
  GuiInt  interpolate_;

public:
  Gradient(const clString& id);
  virtual ~Gradient();

  virtual void execute();

  template <class F> void dispatch_tetvol(F *f);
  template <class F> void dispatch_latticevol(F *f);
};


extern "C" Module* make_Gradient(const clString& id) {
  return new Gradient(id);
}


Gradient::Gradient(const clString& id)
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
  typename F::mesh_type::cell_iterator ci = tvm->cell_begin();
  while (ci != tvm->cell_end())
  {
    result->set_value(f->cell_gradient(*ci), *ci);
    ++ci;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Gradient");
  FieldHandle fh(result);
  ofp->send(fh);
}

template <class F>
void
Gradient::dispatch_latticevol(F *f)
{
  LatVolMeshHandle lvm = f->get_typed_mesh(); 
  LatticeVol<Vector> *result = new LatticeVol<Vector>(lvm, Field::CELL);
  typename F::mesh_type::cell_iterator ci = lvm->cell_begin();
  while (ci != lvm->cell_end())
  {
    Point p;
    lvm->get_center(p, *ci);
    Vector v;
    f->get_gradient(v, p);
    result->set_value(v, *ci);
    ++ci;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Gradient");
  FieldHandle fh(result);
  ofp->send(fh);
}


void
Gradient::execute()
{
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle fieldhandle;
  Field *field;
  if (!(ifp->get(fieldhandle) && (field = fieldhandle.get_rep())))
  {
    return;
  }

  // Create a new Vector field with the same geometry handle as field.
  const string geom_name = field->get_type_name(0);
  const string data_name = field->get_type_name(1);
  if (geom_name == "TetVol")
  {
    if (data_name == "double")
    {
      dispatch_tetvol((TetVol<double> *)field);
    }
    else if (data_name == "int")
    {
      dispatch_tetvol((TetVol<int> *)field);
    }
    else if (data_name == "short")
    {
      dispatch_tetvol((TetVol<short> *)field);
    }
    else if (data_name == "char")
    {
      dispatch_tetvol((TetVol<char> *)field);
    }
    else
    {
      // Don't know what to do with this field type.
      // Signal some sort of error.
    }
  }
  else if (geom_name == "LatticeVol")
  {
    if (data_name == "double")
    {
      dispatch_latticevol((LatticeVol<double> *)field);
    }
    else if (data_name == "int")
    {
      dispatch_latticevol((LatticeVol<int> *)field);
    }
    else if (data_name == "short")
    {
      dispatch_latticevol((LatticeVol<short> *)field);
    }
    else if (data_name == "char")
    {
      dispatch_latticevol((LatticeVol<char> *)field);
    }
    else
    {
      // Don't know what to do with this field type.
      // Signal some sort of error.
    }
  }
  else
  {
    // Don't know what to do with this field type.
    // Signal some sort of error.
    return;
  }

}

} // End namespace SCIRun

