
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Array2.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Heightfield.h>
#include <Packages/rtrt/Core/BrickArray2.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1468
#endif

using rtrt::Array1;
using rtrt::Array2;
using rtrt::Heightfield;
using rtrt::HMCell; 
using rtrt::BrickArray2;

namespace SCIRun {
template<>
void Pio(Piostream& stream, rtrt::Array1<int>& array)
{
  stream.begin_class("rtrtArray1", ARRAY1_RTRT_VERSION);
  int size=array.size();
  Pio(stream, size);
  if(stream.reading()){
    array.remove_all();
    array.grow(size);
  }
  int* obj_arr = array.get_objs();
  if (stream.supports_block_io()) {
    stream.block_io(obj_arr, sizeof(int), size);
  } else {
    for(int i = 0; i < size; i++) {
      Pio(stream, obj_arr[i]);
    }
  }
  stream.end_class();
}

template<>
void Pio(Piostream& stream, rtrt::Array1<float>& array)
{
  stream.begin_class("rtrtArray1", ARRAY1_RTRT_VERSION);
  int size=array.size();
  Pio(stream, size);
  if(stream.reading()){
    array.remove_all();
    array.grow(size);
  }
  float* obj_arr = array.get_objs();
  if (stream.supports_block_io()) {
    stream.block_io(obj_arr, sizeof(float), size);
  } else {
    for(int i = 0; i < size; i++) {
      Pio(stream, obj_arr[i]);
    }
  }
  stream.end_class();
}

template<>
void Pio(Piostream& stream, rtrt::Array1<double>& array)
{
  stream.begin_class("rtrtArray1", ARRAY1_RTRT_VERSION);
  int size=array.size();
  Pio(stream, size);
  if(stream.reading()){
    array.remove_all();
    array.grow(size);
  }
  double* obj_arr = array.get_objs();
  if (stream.supports_block_io()) {
    stream.block_io(obj_arr, sizeof(double), size);
  } else {
    for(int i = 0; i < size; i++) {
      Pio(stream, obj_arr[i]);
    }
  }
  stream.end_class();
}

}

namespace rtrt {
  class Object;
  class VolumeBase;
  class Volume;
  class Material;



template<> void Pio(SCIRun::Piostream& stream, rtrt::Array2<int>& data)
{
  stream.begin_class("rtrtArray2", Array2_VERSION);
  if(stream.reading()){
    // Allocate the array...
    int d1, d2;
    SCIRun::Pio(stream, d1);
    SCIRun::Pio(stream, d2);
    data.resize(d1, d2);
  } else {
    SCIRun::Pio(stream, data.dm1);
    SCIRun::Pio(stream, data.dm2);
  }

  if (stream.supports_block_io()) {
    stream.block_io(data.objs, sizeof(int), data.dm1*data.dm1);
  } else {
    for(int i=0;i<data.dm1;i++){
      for(int j=0;j<data.dm2;j++){
	SCIRun::Pio(stream, data.objs[i][j]);
      }
    }
  }
  stream.end_class();
}

template<> void Pio(SCIRun::Piostream& stream, rtrt::Array2<float>& data)
{
  stream.begin_class("rtrtArray2", Array2_VERSION);
  if(stream.reading()){
    // Allocate the array...
    int d1, d2;
    SCIRun::Pio(stream, d1);
    SCIRun::Pio(stream, d2);
    data.resize(d1, d2);
  } else {
    SCIRun::Pio(stream, data.dm1);
    SCIRun::Pio(stream, data.dm2);
  }

  if (stream.supports_block_io()) {
    stream.block_io(data.objs, sizeof(float), data.dm1*data.dm1);
  } else {
    for(int i=0;i<data.dm1;i++){
      for(int j=0;j<data.dm2;j++){
	SCIRun::Pio(stream, data.objs[i][j]);
      }
    }
  }
  stream.end_class();
}

template<> void Pio(SCIRun::Piostream& stream, rtrt::Array2<double>& data)
{
  stream.begin_class("rtrtArray2", Array2_VERSION);
  if(stream.reading()){
    // Allocate the array...
    int d1, d2;
    SCIRun::Pio(stream, d1);
    SCIRun::Pio(stream, d2);
    data.resize(d1, d2);
  } else {
    SCIRun::Pio(stream, data.dm1);
    SCIRun::Pio(stream, data.dm2);
  }

  if (stream.supports_block_io()) {
    stream.block_io(data.objs, sizeof(double), data.dm1*data.dm1);
  } else {
    for(int i=0;i<data.dm1;i++){
      for(int j=0;j<data.dm2;j++){
	SCIRun::Pio(stream, data.objs[i][j]);
      }
    }
  }
  stream.end_class();
}

}

template struct HMCell<float>;
template class rtrt::BrickArray2<float>;
template class Array2 < HMCell<float> >;
template class Heightfield<BrickArray2<float>,Array2<HMCell<float > > >;





template class Array1<Light*>;

template class Array1<Object*>;

template class Array1<VolumeBase*>;

template class Array1<Volume*>;

#include <Core/Geometry/Point.h>
template class Array1<SCIRun::Point>;

#include <Core/Geometry/Vector.h>
template class Array1<SCIRun::Vector>;

template class Array1<int>;
template class Array1<float>;
template class Array1<double>;

template class Array1<Material*>;
#include <Packages/rtrt/Core/Shadows/ShadowBase.h>
template class Array1<ShadowBase*>;
namespace SCIRun {
template void Pio<rtrt::ShadowBase *>(Piostream &, 
				      rtrt::Array1<rtrt::ShadowBase *> &);
}
#include <Packages/rtrt/Core/Random.h>
#include <Packages/rtrt/Core/HashTable.cc>
#include <Packages/rtrt/Core/HashTableEntry.cc>

template class rtrt::HashTable<rtrt::RandomTable::TableInfo, double*>;
template class rtrt::HashTableEntry<rtrt::RandomTable::TableInfo, double*>;
template class rtrt::HashTable<rtrt::RandomTable::TableInfo, int*>;
template class rtrt::HashTableEntry<rtrt::RandomTable::TableInfo, int*>;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif
