/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
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


#ifndef __OPERATORS_UNARYFIELDOPERATOR_H__
#define __OPERATORS_UNARYFIELDOPERATOR_H__

#include "OperatorThread.h"
#include <Core/Geometry/IntVector.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Uintah/Core/Disclosure/TypeUtils.h>

namespace Uintah {
using namespace SCIRun;

  class UnaryFieldOperator {
  public:
    UnaryFieldOperator(){};
    virtual ~UnaryFieldOperator() {}
    
  protected:
    template<class Field, class ScalarField>
     void initField(Field* field,
		    ScalarField* scalarField);
    
    template<class Field, class ScalarField, class ScalarOp >
     void computeScalars(Field* field,
			 ScalarField* scalarField,
			 ScalarOp op /* ScalarOp should be a functor for
					modiyfying scalars */ );
};

template<class Field, class ScalarField>
void UnaryFieldOperator::initField(Field* field,
				    ScalarField* scalarField)
{
  ASSERT( field->basis_order() == 0 ||
	  field->basis_order() == 1 );

  typename Field::mesh_handle_type mh = field->get_typed_mesh();
  typename ScalarField::mesh_handle_type smh = scalarField->get_typed_mesh();
  BBox box;
  box = smh->get_bounding_box();
  //resize the geometry
  smh->set_ni(mh->get_ni());
  smh->set_nj(mh->get_nj());
  smh->set_nk(mh->get_nk());
  smh->set_transform(mh->get_transform());
  //resize the data storage
  scalarField->resize_fdata();

}

template<class Field, class ScalarField, class Op>
void UnaryFieldOperator::computeScalars(Field* field,
					 ScalarField* scalarField,
					 Op op)
{
  // so far only node and cell centered data
  ASSERT( field->basis_order() == 0 ||
	  field->basis_order() == 1 );


  typename Field::mesh_handle_type mh =
    field->get_typed_mesh();
  typename ScalarField::mesh_handle_type smh =
    scalarField->get_typed_mesh();
 
  if( field->basis_order() == 0){
    typename Field::mesh_type::Cell::iterator it; mh->begin(it);
    typename Field::mesh_type::Cell::iterator end; mh->end(end);
    typename ScalarField::mesh_type::Cell::iterator s_it; smh->begin(s_it);
    for( ; it != end; ++it, ++s_it){
      scalarField->fdata()[*s_it] = op(field->fdata()[*it]);
    }
  } else {
    typename Field::mesh_type::Node::iterator it; mh->begin(it);
    typename Field::mesh_type::Node::iterator end; mh->end(end);
    typename ScalarField::mesh_type::Node::iterator s_it; smh->begin(s_it);
    
    for( ; it != end; ++it, ++s_it){
      scalarField->fdata()[*s_it] = op(field->fdata()[*it]);
    }
  }  
}

} // End namespace Uintah
#endif // __OPERATORS_UNARYFIELDOPERATOR_H__


