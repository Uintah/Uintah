/*
 *  TensorField.cc: Data structure to represent tensor fields
 *  ---------------
 *
 *  This file was created initally by duping the sciBoolean file
 *  so as I know more about what the heck everything is I'll 
 *  try to through some comments in here.
 *
 *  Eric Lundberg - 1998
 *  
 */

#include <Datatypes/TensorField.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

static Persistent* maker()
{
    return scinew TensorField<short>(0,0,0);
}

template<class DATA>
PersistentTypeID TensorField<DATA>::type_id("TensorField", "TensorFieldBase", maker);

/* The default constructor...currently not taking any args...*/
template<class DATA>
TensorField<DATA>::TensorField(int in_slices, int in_width, int in_height)
{
  m_slices = in_slices;
  m_width = in_width;
  m_height = in_height;
  
  /* Set up the arrays for the tensors*/
  m_tensor_field.resize(TENSOR_ELEMENTS);
  
  for (int ii = 0; ii < TENSOR_ELEMENTS; ii ++)
    m_tensor_field[ii].newsize(m_slices, m_width, m_height);
  
  /* Set up the arrays for the eigenvectors*/
  m_e_vectors.resize(EVECTOR_ELEMENTS);
  
  /* Set up the arrays for the eigenvalues*/
  m_e_values.resize(EVECTOR_ELEMENTS);

  m_tensorsGood = m_vectorsGood = m_valuesGood = 0;
}

/*note all index values should be from 0 - (value)*/
template<class DATA>
int TensorField<DATA>::AddSlice(int in_slice, int in_tensor_component, FILE* in_file)
{
  if (in_slice > m_slices)
    return 0;

  fread(&(m_tensor_field[in_tensor_component](in_slice,0,0)), sizeof(DATA), m_height*m_width, in_file);
  
  /*for (int yy = 0; yy < m_height; yy++)
    for (int xx = 0; xx < m_width; xx++)
    if (fread(&(m_tensor_field[in_tensor_component](in_slice,yy,xx)), sizeof(DATA), 1, in_file) != 1) 
	{	
	printf("TensorComponent file wasn't big enough");
	return 0;
	}*/
  return 1;
}

/* Constructor used to clone objects...either through a previous
   TensorFields .clone call, or explicitly with scinew TensorField(TFtoCopy)
   as such this function should actually copy all the data in t.x to this.x
   in a deep fashion */
template<class DATA>
TensorField<DATA>::TensorField(const TensorField<DATA>& t)
{
  /*NOTE - IMPEMENT ME!*/
    NOT_FINISHED("TensorField copy ctor\n");
}

/*Destructor*/
template<class DATA>
TensorField<DATA>::~TensorField()
{ 
}

/*returns a totally different, but logically equivalent version of 'this'*/
template<class DATA>
TensorField<DATA>* TensorField<DATA>::clone() const
{
    return scinew TensorField<DATA>(*this);
}

/* Set what version of the tensorfield we are currently working with - so we
   can support old verions of this datatype on the IO ports */
#define TENSORFIELD_VERSION 1

/* And do the actual IO stuff....*/
template<class DATA>
void TensorField<DATA>::io(Piostream& stream)
{
    stream.begin_class("TensorField", TENSORFIELD_VERSION);
    
    Pio(stream, m_type);
    Pio(stream, m_slices);
    Pio(stream, m_width);
    Pio(stream, m_height);
  
    printf("Hello world\n");
    /* Set up the arrays for the tensors*/
    m_tensor_field.resize(TENSOR_ELEMENTS);
    
    for (int ii = 0; ii < TENSOR_ELEMENTS; ii ++)
      m_tensor_field[ii].newsize(m_slices, m_width, m_height);
    
    /* Set up the arrays for the eigenvectors*/
    m_e_vectors.resize(EVECTOR_ELEMENTS);
    /* Set up the arrays for the eigenvalues*/
    m_e_values.resize(EVECTOR_ELEMENTS);


    for (int slice = 0; slice < TENSOR_ELEMENTS; slice++)
      Pio(stream, m_tensor_field[slice]);
    stream.end_class();
}
