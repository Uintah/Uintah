/* TensorFieldBase.cc
   ------------------
   
   This is the implementation of said dummy class as
   described in the header file.

   Eric Lundberg, 10/8/1998
 
   */

#include <Packages/DaveW/Core/Datatypes/General/TensorFieldBase.h>
#include <Core/Containers/String.h>
#include <Core/Util/NotFinished.h>

namespace DaveW {
PersistentTypeID TensorFieldBase::type_id("TensorFieldBase", "Datatype", 0);


TensorFieldBase::TensorFieldBase()
{
}

TensorFieldBase::TensorFieldBase(const TensorFieldBase& /*in_tfb*/)
{
    NOT_FINISHED("TensorFieldBase copy ctor");
}

TensorFieldBase::~TensorFieldBase()
{
}

#define TENSORFIELDBASE_VERSION 1

void TensorFieldBase::io(Piostream& stream)
{
using namespace SCIRun;

    stream.begin_class("TensorFieldBase", TENSORFIELDBASE_VERSION);
    Pio(stream, m_type);
    Pio(stream, bmin);
    Pio(stream, bmax);
    Pio(stream, diagonal);
    Pio(stream, m_slices);
    Pio(stream, m_width);
    Pio(stream, m_height);
    Pio(stream, m_tensorsGood);
    Pio(stream, m_vectorsGood);
    if (m_vectorsGood)
      {
	Pio(stream, m_e_vectors);
      }

    /* and the scalar fields if we have them */
    Pio(stream, m_valuesGood);
    if (m_valuesGood)
      {
	Pio(stream, m_e_values);
      }
    Pio(stream, m_inside);
}

void TensorFieldBase::set_type(int in_type)
{
  m_type = in_type;
}
int TensorFieldBase::get_type(void)
{
  return m_type;
}

void TensorFieldBase::get_bounds(Point &min, Point &max) {
    min = bmin;
    max = bmax;
}

void TensorFieldBase::set_bounds(const Point& min, const Point& max) {
    bmin=min;
    bmax=max;
    diagonal=bmax-bmin;
    if (m_vectorsGood)
	for (int i=0; i<EVECTOR_ELEMENTS; i++) 
	    m_e_vectors[i].set_bounds(min, max);
    if (m_valuesGood)
	for (int i=0; i<EVECTOR_ELEMENTS; i++) 
	    m_e_values[i].set_bounds(min, max);
}
} // End namespace DaveW


