/*
 *  TensorField.h
 *
 *  Eric Lundberg 1998
 */

#ifndef SCI_Datatypes_TensorField_h
#define SCI_Datatypes_TensorField_h 1

#include <stdio.h>
#include <Datatypes/Datatype.h>
#include <Datatypes/TensorFieldBase.h>
#include <Datatypes/VectorFieldRG.h>
#include <Datatypes/ScalarFieldRGdouble.h>
#include <Classlib/LockingHandle.h>
#include <Classlib/Array3.h>
#include <Classlib/Array2.h>
#include <Classlib/Array1.h>


#define TENSOR_ELEMENTS 6 /*Number of elements in the 3x3 tensor we car about*/
#define EVECTOR_ELEMENTS 3 /*Number of eigen vectors produced by the tensor matrix*/

template<class DATA>
class TensorField : public TensorFieldBase {
public:
    int m_slices;
    int m_width;
    int m_height;
    Array1< Array3<DATA> > m_tensor_field;
    Array1< VectorFieldRG* > m_e_vectors;
    Array1< ScalarFieldRGdouble* > m_e_values;
    int m_tensorsGood;
    int m_vectorsGood;
    int m_valuesGood;

    /*Functions*/
    int AddSlice(int in_slice, int in_tensor_component, FILE* in_file);
    
    Array1<Array3<DATA> > * getData(void){return &m_tensor_field;}


    TensorField(int in_slices, int in_width, int in_height); /*Default Constructor*/
    TensorField(const TensorField&); /*Deep Copy Constructor*/

    virtual ~TensorField(); /*Destructor*/
#ifdef __GNUG__
    virtual TensorField<DATA>* clone() const; /*makes a real copy of this*/
#else
    virtual TensorFieldBase* clone() const; /*makes a real copy of this*/
#endif

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
