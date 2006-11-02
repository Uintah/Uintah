/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldInterfaceAux.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Thread/Mutex.h>
#include <map>

namespace SCIRun{

// initialize the static member type_id
PersistentTypeID Field::type_id("Field", "PropertyManager", 0);

// A list to keep a record of all the different Field types that
// are supported through a virtual interface
Mutex FieldTypeIDMutex("Field Type ID Table Lock");
static std::map<string,FieldTypeID*>* FieldTypeIDTable = 0;

FieldTypeID::FieldTypeID(const string&type,
                         FieldHandle (*field_maker)(),
                         FieldHandle (*field_maker_mesh)(MeshHandle)) :
    type(type),
    field_maker(field_maker),
    field_maker_mesh(field_maker_mesh)
{
  FieldTypeIDMutex.lock();
  if (FieldTypeIDTable == 0)
  {
    FieldTypeIDTable = scinew std::map<string,FieldTypeID*>;
  }
  else
  {
    map<string,FieldTypeID*>::iterator dummy;
    
    dummy = FieldTypeIDTable->find(type);
    
    if (dummy != FieldTypeIDTable->end())
    {
      if (((*dummy).second->field_maker != field_maker) ||
          ((*dummy).second->field_maker_mesh != field_maker_mesh))
      {
        std::cerr << "WARNING: duplicate field type exists: " << type << "\n";
        FieldTypeIDMutex.unlock();
        return;
      }
    }
  }
  std::cout << "Adding FieldTypeId :"<<type<<"\n";
  
  (*FieldTypeIDTable)[type] = this;
  FieldTypeIDMutex.unlock();
}


FieldHandle
Create_Field(string type, MeshHandle mesh)
{
  FieldHandle handle(0);
  FieldTypeIDMutex.lock();
  std::map<string,FieldTypeID*>::iterator it;
  it = FieldTypeIDTable->find(type);
  if (it != FieldTypeIDTable->end()) 
  {
    handle = (*it).second->field_maker_mesh(mesh);
  }
  FieldTypeIDMutex.unlock();
  return (handle);
}

FieldHandle
Create_Field(string type)
{
  FieldHandle handle(0);
  FieldTypeIDMutex.lock();
  std::map<string,FieldTypeID*>::iterator it;
  it = FieldTypeIDTable->find(type);
  if (it != FieldTypeIDTable->end()) 
  {
    handle = (*it).second->field_maker();
  }
  FieldTypeIDMutex.unlock();
  return (handle);
}

Field::Field()
{
}

Field::~Field()
{
}

const int FIELD_VERSION = 3;

void 
Field::io(Piostream& stream)
{
  int version = stream.begin_class("Field", FIELD_VERSION);
  if (version < 2) {
    // The following was FIELD_VERSION 1 data_at ordering
    //     enum data_location{
    //       NODE,
    //       EDGE,
    //       FACE,
    //       CELL,
    //       NONE
    //     };

    unsigned int tmp;
    int order = 999;
    Pio(stream, tmp);
    if (tmp == 0) {
      // data_at_ was NODE
      order = 1;
      if (mesh()->dimensionality() == 0) order = 0;
    } else if (tmp == 4) {
      // data_at_ was NONE
      order = -1;
    } else {
      // data_at_ was somewhere else
      order = 0;
    }
    
    if (order != basis_order()) {
      // signal error in the stream and return;
      stream.flag_error();
      return;
    }
  } else if (version < 3) {
    int order;
    Pio(stream, order);
    if (order != basis_order()) {
      // signal error in the stream and return;
      stream.flag_error();
      return;
    }
  }

  PropertyManager::io(stream);
  stream.end_class();
}

bool 
Field::has_virtual_interface()
{
  return (false);
}


void
Field::resize_fdata()
{
  ASSERTFAIL("Field interface has no resize_fdata() function");
}


void 
Field::get_value(char &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}

void 
Field::get_value(unsigned char &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}

void 
Field::get_value(short &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}

void 
Field::get_value(unsigned short &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}

void 
Field::get_value(int &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}

void 
Field::get_value(unsigned int &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}

void 
Field::get_value(long &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}

void 
Field::get_value(unsigned long &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}
void 
Field::get_value(long long &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}

void 
Field::get_value(unsigned long long &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}

void 
Field::get_value(float &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}

void 
Field::get_value(double &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}

void 
Field::get_value(Vector &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}

void 
Field::get_value(Tensor &val, Mesh::index_type i) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for get_value");
}



void 
Field::set_value(const char &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}

void 
Field::set_value(const unsigned char &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}

void 
Field::set_value(const short &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}

void 
Field::set_value(const unsigned short &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}

void 
Field::set_value(const int &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}

void 
Field::set_value(const unsigned int &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}

void 
Field::set_value(const long &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}

void 
Field::set_value(const unsigned long &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}

void 
Field::set_value(const long long &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}

void 
Field::set_value(const unsigned long long &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}

void 
Field::set_value(const float &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}

void 
Field::set_value(const double &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}

void 
Field::set_value(const Vector &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}

void 
Field::set_value(const Tensor &val, Mesh::index_type i)
{
  ASSERTFAIL("Field interface has no virtual function implementation for set_value");
}


void 
Field::interpolate(char &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}

void 
Field::interpolate(unsigned char &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}

void 
Field::interpolate(short &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}

void 
Field::interpolate(unsigned short &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}

void 
Field::interpolate(int &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}

void 
Field::interpolate(unsigned int &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}

void 
Field::interpolate(long &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}

void 
Field::interpolate(unsigned long &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}

void 
Field::interpolate(long long &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}

void 
Field::interpolate(unsigned long long &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}

void 
Field::interpolate(float &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}

void 
Field::interpolate(double &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}

void 
Field::interpolate(Vector &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}

void 
Field::interpolate(Tensor &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for interpolate");
}


void 
Field::gradient(vector<char> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}

void 
Field::gradient(vector<unsigned char> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}

void 
Field::gradient(vector<short> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}

void 
Field::gradient(vector<unsigned short> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}

void 
Field::gradient(vector<int> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}

void 
Field::gradient(vector<unsigned int> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}

void 
Field::gradient(vector<long> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}

void 
Field::gradient(vector<unsigned long> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}

void 
Field::gradient(vector<long long> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}

void 
Field::gradient(vector<unsigned long long> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}

void 
Field::gradient(vector<float> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}

void 
Field::gradient(vector<double> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}

void 
Field::gradient(vector<Vector> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}

void 
Field::gradient(vector<Tensor> &val, const vector<double> &coords, Mesh::index_type elem_idx) const
{
  ASSERTFAIL("Field interface has no virtual function implementation for gradient");
}


// Additional definitions for the FieldInterface classes

ScalarFieldInterfaceHandle
Field::query_scalar_interface(ProgressReporter *reporter)
{
  if (basis_order() == -1) { return 0; }

  const TypeDescription *ftd = get_type_description();
  const TypeDescription *ltd = order_type_description();
  CompileInfoHandle ci = ScalarFieldInterfaceMaker::get_compile_info(ftd, ltd);
  LockingHandle<ScalarFieldInterfaceMaker> algo(0);
  
  ProgressReporter my_reporter;
  if (!reporter) reporter = &my_reporter;
  if ( DynamicCompilation::compile( ci, algo, true, reporter ) )
    return algo->make(this);
  else
    return 0;
}


VectorFieldInterfaceHandle
Field::query_vector_interface(ProgressReporter *reporter)
{
  if (basis_order() == -1) { return 0; }
  const TypeDescription *ftd = get_type_description();
  const TypeDescription *ltd = order_type_description();
  CompileInfoHandle ci = VectorFieldInterfaceMaker::get_compile_info(ftd, ltd);
  LockingHandle<VectorFieldInterfaceMaker> algo(0);
  
  ProgressReporter my_reporter;
  if (!reporter) reporter = &my_reporter;
  if ( DynamicCompilation::compile( ci, algo, true, reporter ) )
    return algo->make(this);
  else
    return 0;
}


TensorFieldInterfaceHandle
Field::query_tensor_interface(ProgressReporter *reporter)
{
  if (basis_order() == -1) { return 0; }

  const TypeDescription *ftd = get_type_description();
  const TypeDescription *ltd = order_type_description();
  CompileInfoHandle ci = TensorFieldInterfaceMaker::get_compile_info(ftd, ltd);
  LockingHandle<TensorFieldInterfaceMaker> algo(0);
  
  ProgressReporter my_reporter;
  if (!reporter) reporter = &my_reporter;
  if ( DynamicCompilation::compile( ci, algo, true, reporter ) )
    return algo->make(this);
  else
    return 0;
}

}
