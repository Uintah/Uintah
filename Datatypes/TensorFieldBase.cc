/* TensorFieldBase.cc
   ------------------
   
   This is the implementation of said dummy class as
   described in the header file.

   Eric Lundberg, 10/8/1998
 
   */

#include <Datatypes/TensorFieldBase.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

PersistentTypeID TensorFieldBase::type_id("TensorFieldBase", "Datatype", 0);


TensorFieldBase::TensorFieldBase()
{
}

TensorFieldBase::TensorFieldBase(const TensorFieldBase& in_tfb)
{
    NOT_FINISHED("TensorFieldBase copy ctor");
}

TensorFieldBase::~TensorFieldBase()
{
}

void TensorFieldBase::io(Piostream& stream)
{
    NOT_FINISHED("TensorFieldBase::io");
}

void TensorFieldBase::set_type(int in_type)
{
  m_type = in_type;
}
int TensorFieldBase::get_type(void)
{
  return m_type;
}
