/*
 *  TensorField.h
 *
 *  Eric Lundberg 1998
 */

#ifndef SCI_DaveW_Datatypes_TensorField_h
#define SCI_DaveW_Datatypes_TensorField_h 1

#include <DaveW/Datatypes/General/TensorFieldBase.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Datatypes/Datatype.h>
#include <stdio.h>

namespace DaveW {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array1;
using SCICore::Containers::Array2;
using SCICore::Containers::Array3;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

using namespace SCICore::Datatypes;

template<class DATA>
class TensorField : public TensorFieldBase {
public:
    Array1< Array3<DATA> > m_tensor_field;

    /*Functions*/
    int AddSlice(int in_slice, int in_tensor_component, FILE* in_file);
    
    Array1<Array3<DATA> > * getData(void){return &m_tensor_field;}


    TensorField(int in_slices, int in_width, int in_height); /*Default Constructor*/
    TensorField(int in_slices, int in_width, int in_height, float *data);
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

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/09/01 05:27:37  dmw
// more DaveW datatypes...
//

#endif /* SCI_DaveW_Datatypes_TensorField_h */
