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
#include <Classlib/LockingHandle.h>
#include <Classlib/Array3.h>
#include <Classlib/Array2.h>
#include <Classlib/Array1.h>


template<class DATA>
class TensorField : public TensorFieldBase {
public:
    Array1< Array3<DATA> > m_tensor_field;

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

    virtual int interpolate(const Point&, double[][3], int&, int=0);
    virtual int interpolate(const Point&, double[][3]);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
