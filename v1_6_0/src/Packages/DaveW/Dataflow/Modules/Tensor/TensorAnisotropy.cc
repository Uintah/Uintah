#include <Packages/DaveW/Core/Datatypes/General/TensorField.h>
#include <Packages/DaveW/Core/Datatypes/General/TensorFieldPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Datatypes/VectorField.h>
#include <Core/Malloc/Allocator.h>

namespace DaveW {
using namespace SCIRun;
using namespace DaveW;

class TensorAnisotropy : public Module 
{ 
  TensorFieldIPort * i_tensor;
  TensorFieldOPort * o_tensor;
  ScalarFieldOPort * o_scal_line;
  ScalarFieldOPort * o_scal_plane;
  ScalarFieldOPort * o_scal_index;
  ScalarFieldOPort * o_scal_relative;
  ScalarFieldOPort * o_scal_fractional;
  ScalarFieldOPort * o_scal_1_volume_ratio;

  ScalarFieldRGdouble * sf_line;
  ScalarFieldRGdouble * sf_plane;
  ScalarFieldRGdouble * sf_index;

  ScalarFieldRGdouble * sf_relative;
  ScalarFieldRGdouble * sf_fractional;
  ScalarFieldRGdouble * sf_1_volume_ratio;

public: 
  TensorAnisotropy(const clString& id);
  virtual ~TensorAnisotropy();
  virtual void execute();

  /*Functions*/
  template <class DATA>
    void do_it(TensorField<DATA> *tensor_base);

private:
  TensorFieldBase *m_tf;

  int make_sf(ScalarFieldRGdouble* &outField);
  template <class DATA> inline
    void set_value(Array1<Array3<DATA> > *in_data, short x, short y, short slice, short tensor, DATA val);
  template <class DATA> inline
    DATA get_value(Array1<Array3<DATA> > *in_data, short x, short y, short slice, short tensor);
};

extern "C" Module* make_TensorAnisotropy(const clString& id) { 
    return new TensorAnisotropy(id); 
} 
  
//--------------------------------------------------------------- 
TensorAnisotropy::TensorAnisotropy(const clString& id) 
: Module("TensorAnisotropy", id, Filter)
{
  /*TensorPorts*/
  i_tensor = scinew TensorFieldIPort(this, "TensorField", TensorFieldIPort::Atomic);
  add_iport(i_tensor);

  o_tensor = scinew TensorFieldOPort(this, "TensorField", TensorFieldIPort::Atomic);
  add_oport(o_tensor);

  /* Westin - Harvard */
  o_scal_line = scinew ScalarFieldOPort(this, "Linear Anisotropy", ScalarFieldIPort::Atomic);
  add_oport(o_scal_line);

  o_scal_plane = scinew ScalarFieldOPort(this, "Planar Anisotropy", ScalarFieldIPort::Atomic);
  add_oport(o_scal_plane);

  o_scal_index = scinew ScalarFieldOPort(this, "Anisotropy Index", ScalarFieldIPort::Atomic);
  add_oport(o_scal_index);

  /* Basser - NIH */
  o_scal_relative = scinew ScalarFieldOPort(this, "Relative Anisotropy", ScalarFieldIPort::Atomic);
  add_oport(o_scal_relative);

  o_scal_fractional = scinew ScalarFieldOPort(this, "Fractional Anisotropy", ScalarFieldIPort::Atomic);
  add_oport(o_scal_fractional);

  o_scal_1_volume_ratio = scinew ScalarFieldOPort(this, "1 - Volume Ratio", ScalarFieldIPort::Atomic);
  add_oport(o_scal_1_volume_ratio);

  sf_line = sf_plane = sf_index = sf_relative = sf_fractional = sf_1_volume_ratio = NULL;
}

//------------------------------------------------------------ 
TensorAnisotropy::~TensorAnisotropy()
{
  /*Nothing to destroy*/
} 

//-------------------------------------------------------------- 
void TensorAnisotropy::execute() 
{ 
  TensorFieldHandle tf_handle;
  TensorFieldBase *tensor_field;

  if (!i_tensor->get(tf_handle)) return;
  tensor_field = tf_handle.get_rep();
  if (tensor_field != NULL)
    printf("got something that thinks it's a tensor field\n");
  else
    printf("null tf\n");
  
  /*Ok we have the tensorfile, so set our modules tf to it*/
  m_tf = tensor_field;

  switch(tensor_field->get_type())
    {
    case CHAR:
      do_it((TensorField<char>*)tensor_field);
      break;
    case UCHAR:
      do_it((TensorField<unsigned char>*)tensor_field);
      break;
    case SHORT:
      do_it((TensorField<short>*)tensor_field);
      break;
    case USHORT:
      do_it((TensorField<unsigned short>*)tensor_field);
      break;
    case INT:
      do_it((TensorField<int>*)tensor_field);
      break;
    case UINT:
      do_it((TensorField<unsigned int>*)tensor_field);
      break;
    case LONG:
      do_it((TensorField<long>*)tensor_field);
      break;
    case ULONG:
      do_it((TensorField<unsigned long>*)tensor_field);
      break;
    case FLOAT:
      do_it((TensorField<float>*)tensor_field);
      break;
    case DOUBLE:
      do_it((TensorField<double>*)tensor_field);
      break;
    default:
       printf("TensorAnisotropy doesn't know how to deal with this type. This means that either a bad type got set in the base,or that a new type was added to the available types in the TensorField and no one told us about it.\n");
    }

  /* Send output here */
  o_tensor->send(*scinew TensorFieldHandle(tensor_field));
  o_scal_line->send(*scinew ScalarFieldHandle(sf_line));
  o_scal_plane->send(*scinew ScalarFieldHandle(sf_plane));
  o_scal_index->send(*scinew ScalarFieldHandle(sf_index));
  o_scal_relative->send(*scinew ScalarFieldHandle(sf_relative));
  o_scal_fractional->send(*scinew ScalarFieldHandle(sf_fractional));
  o_scal_1_volume_ratio->send(*scinew ScalarFieldHandle(sf_1_volume_ratio));
} 
  


/***************************************************
 PRIVATE FUNCTIO
NS - NO TRESSPASSING! THIS MEANS YOU!
 ***************************************************/
int TensorAnisotropy::make_sf(ScalarFieldRGdouble* &outField)
{
  if (outField != NULL)
    return 0;
  
  Point min, max;
  m_tf->get_bounds(min,max);
 

  outField = scinew ScalarFieldRGdouble();
  if (outField == NULL)
    return 0; 

  /* now activate the boundry */
  outField->set_bounds(min, max);
  outField->resize(m_tf->m_height, m_tf->m_width, m_tf->m_slices);
  outField->grid.initialize(0);

  printf("made\n");
  return 1; //all good baby!
}

template <class DATA>
void TensorAnisotropy::do_it(TensorField<DATA> *tensor_base)
{
  double e1, e2, e3, average;
  double sum_squared_ave_diff, sum_squared;
  double sqrt_six = sqrt(6.0);

  if ( !make_sf(sf_line) ||
       !make_sf(sf_plane) ||
       !make_sf(sf_index) ||
       !make_sf(sf_relative) ||
       !make_sf(sf_fractional) ||
       !make_sf(sf_1_volume_ratio) )
    {
      fprintf(stderr, "Failed to create at least on of the output fields! No computation for YOU! :)\n");
      return;
    }
  
  /* Create us some data foo! */
  
  for (int z = 0; z < m_tf->m_slices; z++) 
    {
      for (int y = 0; y < m_tf->m_width; y++) 
	{
	  for (int x = 0; x < m_tf->m_height; x++)
	    {
	      e1 = tensor_base->m_e_values[0].grid(x,y,z);
	      e2 = tensor_base->m_e_values[1].grid(x,y,z);
	      e3 = tensor_base->m_e_values[2].grid(x,y,z);
	      float inside = tensor_base->m_inside(x,y,z);
	      average = (e1 + e2 + e3)/3.0;
	      if (average < 0.00000001)
		{
		  printf("hack: %f\n", average);
		  average = 0.00000001;
		}
	      sf_line->grid(x,y,z) = (e1 - e3)/(3.0 * average)*inside;
	      sf_plane->grid(x,y,z) = (2.0*(e2 - e3))/(3.0 * average)*inside;
	      sf_index->grid(x,y,z) = sf_line->grid(x,y,z) + sf_plane->grid(x,y,z)*inside;

	      sum_squared_ave_diff = ((e1 - average) * (e1 - average)) + 
		                     ((e2 - average) * (e2 - average)) + 
		                     ((e3 - average) * (e3 - average));
	      sum_squared = e1 * e1 + e2 * e2 + e3 * e3;
  
	      sf_relative->grid(x,y,z) = (sqrt(sum_squared_ave_diff))/(sqrt_six * average);
	      sf_fractional->grid(x,y,z) = sqrt( (3.0 * sum_squared_ave_diff)/(2.0 * sum_squared) );
	      sf_1_volume_ratio->grid(x,y,z) = 1.0 - ((e1*e2*e3)/(average*average*average));
	    }
	}
    }
} // End namespace DaveW
}

