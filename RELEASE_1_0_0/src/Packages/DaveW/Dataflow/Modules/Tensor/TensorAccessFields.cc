#include <Packages/DaveW/Core/Datatypes/General/TensorField.h>
#include <Packages/DaveW/Core/Datatypes/General/TensorFieldPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Datatypes/VectorField.h>
#include <Core/Malloc/Allocator.h>

#define TENSOR_CONFIG "tensor_cfg"  

namespace DaveW {
using namespace SCIRun;
using namespace DaveW;

class TensorAccessFields : public Module 
{ 
  TensorFieldIPort * i_tensor;
  TensorFieldOPort * o_tensor;
  VectorFieldOPort * o_vect0;
  VectorFieldOPort * o_vect1;
  VectorFieldOPort * o_vect2;
  ScalarFieldOPort * o_scal0;
  ScalarFieldOPort * o_scal1;
  ScalarFieldOPort * o_scal2;

public: 
  TensorAccessFields(const clString& id); 
  virtual ~TensorAccessFields(); 
  virtual void execute(); 
};

template <class DATA>
void set_value(Array1<Array3<DATA> > in_data, short x, short y, short slice, short tensor, short val);

template <class DATA>
DATA get_value(Array1<Array3<DATA> > in_data, short x, short y, short slice, short tensor);


extern "C" Module* make_TensorAccessFields(const clString& id) { 
    return new TensorAccessFields(id); 
} 
  
int width, num_slices;

//--------------------------------------------------------------- 
TensorAccessFields::TensorAccessFields(const clString& id) 
: Module("TensorAccessFields", id, Filter)
{
  i_tensor = scinew TensorFieldIPort(this, "TensorField", TensorFieldIPort::Atomic);
  add_iport(i_tensor);

  o_tensor = scinew TensorFieldOPort(this, "TensorField + Eigen Vectors/Values", TensorFieldIPort::Atomic);
  add_oport(o_tensor);

  o_vect0 = scinew VectorFieldOPort(this, "VectorField - princible components", VectorFieldIPort::Atomic);
  add_oport(o_vect0);

  o_vect1 = scinew VectorFieldOPort(this, "VectorField - secondary components", VectorFieldIPort::Atomic);
  add_oport(o_vect1);

  o_vect2 = scinew VectorFieldOPort(this, "VectorField - tertiary components", VectorFieldIPort::Atomic);
  add_oport(o_vect2);

  o_scal0 = scinew ScalarFieldOPort(this, "ScalarField + quantized length of princible components", ScalarFieldIPort::Atomic);
  add_oport(o_scal0);

  o_scal1 = scinew ScalarFieldOPort(this, "ScalarField + quantized length of princible components", ScalarFieldIPort::Atomic);
  add_oport(o_scal1);

  o_scal2 = scinew ScalarFieldOPort(this, "ScalarField + quantized length of princible components", ScalarFieldIPort::Atomic);
  add_oport(o_scal2);
} 

//------------------------------------------------------------ 
TensorAccessFields::~TensorAccessFields()
{
  /*Nothing to destroy*/
} 

//-------------------------------------------------------------- 
void TensorAccessFields::execute() 
{ 
  TensorFieldHandle tf_handle;
  TensorFieldBase *tensor_field;

  if (!i_tensor->get(tf_handle)) return;
  
  tensor_field = tf_handle.get_rep();
  if (tensor_field != NULL)
    printf("got something that thinks it's a tensor field\n");
  else
    printf("null tf\n");

  //We just passing the data along in other scirun friendly formats....
  o_tensor->send(*scinew TensorFieldHandle(tensor_field));
  /*Note shouldn't matter what 'type' we make the tensorfield in these casts, we just need access to the member data*/
  o_vect0->send(*scinew VectorFieldHandle( &((TensorField<float>*)tensor_field)->m_e_vectors[0]));
  o_vect1->send(*scinew VectorFieldHandle( &((TensorField<float>*)tensor_field)->m_e_vectors[1]));
  o_vect2->send(*scinew VectorFieldHandle( &((TensorField<float>*)tensor_field)->m_e_vectors[2]));
  o_scal0->send(*scinew ScalarFieldHandle( &((TensorField<float>*)tensor_field)->m_e_values[0]));
  o_scal1->send(*scinew ScalarFieldHandle( &((TensorField<float>*)tensor_field)->m_e_values[1]));
  o_scal2->send(*scinew ScalarFieldHandle( &((TensorField<float>*)tensor_field)->m_e_values[2]));

} // End namespace DaveW
} 

