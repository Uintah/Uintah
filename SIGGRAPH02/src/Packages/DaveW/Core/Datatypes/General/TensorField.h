/*
 *  TensorField.h
 *
 *  Eric Lundberg 1998
 */

#ifndef SCI_Packages_DaveW_Datatypes_TensorField_h
#define SCI_Packages_DaveW_Datatypes_TensorField_h 1

#include <Packages/DaveW/Core/Datatypes/General/TensorFieldBase.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Containers/Array3.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Datatype.h>
#include <stdio.h>

namespace DaveW {
using namespace SCIRun;


template<class DATA>
class TensorField : public TensorFieldBase {
public:
    Array1< Array3<DATA> > m_tensor_field;

    /*Functions*/
    int AddSlice(int in_slice, int in_tensor_component, FILE* in_file);
    
    Array1<Array3<DATA> > * getData(void){return &m_tensor_field;}


    TensorField(int in_slices, int in_width, int in_height); /*Default Constructor*/
    TensorField(int in_slices, int in_width, int in_height, float *data, int just_tensors=0);
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
} // End namespace DaveW



#endif /* SCI_Packages/DaveW_Datatypes_TensorField_h */
