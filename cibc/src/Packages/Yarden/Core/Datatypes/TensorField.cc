
/*
 *  Tensor.cc
 *
 *  Written by:
 *   Author: Packages/Yarden Livnat
 *   
 *   Department of Computer Science
 *   University of Utah
 *   Date: Oct 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Packages/Yarden/Core/Datatypes/TensorField.h>

namespace Yarden {
using namespace SCIRun;
    //    using LockingHandle;

    PersistentTypeID TensorFieldBase::type_id("TensorFieldBase", 
	    "Datatype", 0 );

    template<class T>
    Persistent* make_TensorField() {
      return scinew TensorField<T>(0,0,0);
    }

    PersistentTypeID TensorField< SymTensor<float,3> >::
    type_id("TensorField<SymTensor<float,3>>", 
	    "TensorFieldBase", 
	    make_TensorField< SymTensor<float,3> >);

    PersistentTypeID TensorField< SymTensor<unsigned char,3> >::
    type_id("TensorField<SymTensor<uchar,3>>", 
	    "TensorFieldBase", 
	    make_TensorField< SymTensor<unsigned char,3> >);

    PersistentTypeID TensorField< SymTensor<char,3> >::
    type_id("TensorField<SymTensor<char,3>>", 
	    "TensorFieldBase", 
	    make_TensorField< SymTensor<char,3> >);

} // End namespace Yarden

