
/*
 *  ScalarFieldRG.cc: Templated Scalar Fields defined on a Regular grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2000
 *
 *  Copyright (C) 2000 SCI Group
 *
 */


#include <Core/Datatypes/ScalarFieldRG.h>

using namespace SCIRun;


template <>
ScalarFieldRGT<double>::ScalarFieldRGT()
  : ScalarFieldRGBase("double")
{
}

template <>
ScalarFieldRGT<float>::ScalarFieldRGT()
  : ScalarFieldRGBase("float")
{
}

template <>
ScalarFieldRGT<int>::ScalarFieldRGT()
  : ScalarFieldRGBase("int")
{
}

template <>
ScalarFieldRGT<short>::ScalarFieldRGT()
  : ScalarFieldRGBase("short")
{
}

template <>
ScalarFieldRGT<unsigned short>::ScalarFieldRGT()
  : ScalarFieldRGBase("unsigned short")
{
}

template <>
ScalarFieldRGT<char>::ScalarFieldRGT()
  : ScalarFieldRGBase("char")
{
}

template <>
ScalarFieldRGT<unsigned char>::ScalarFieldRGT()
  : ScalarFieldRGBase("unsigned char")
{
}


Persistent* ScalarFieldRGmaker_double()
{
  return scinew ScalarFieldRGT<double>;
}

Persistent* ScalarFieldRGmaker_float()
{
  return scinew ScalarFieldRGT<float>;
}

Persistent* ScalarFieldRGmaker_int()
{
  return scinew ScalarFieldRGT<int>;
}

Persistent* ScalarFieldRGmaker_short()
{
  return scinew ScalarFieldRGT<short>;
}

Persistent* ScalarFieldRGmaker_ushort()
{
  return scinew ScalarFieldRGT<unsigned short>;
}

Persistent* ScalarFieldRGmaker_char()
{
  return scinew ScalarFieldRGT<char>;
}

Persistent* ScalarFieldRGmaker_uchar()
{
  return scinew ScalarFieldRGT<unsigned char>;
}


template <>
PersistentTypeID ScalarFieldRGT<double>::type_id("ScalarFieldRGdouble", "ScalarField", ScalarFieldRGmaker_double);

template <>
PersistentTypeID ScalarFieldRGT<float>::type_id("ScalarFieldRGfloat", "ScalarField", ScalarFieldRGmaker_float);

template <>
PersistentTypeID ScalarFieldRGT<int>::type_id("ScalarFieldRGint", "ScalarField", ScalarFieldRGmaker_int);

template <>
PersistentTypeID ScalarFieldRGT<short>::type_id("ScalarFieldRGshort", "ScalarField", ScalarFieldRGmaker_short);

template <>
PersistentTypeID ScalarFieldRGT<ushort>::type_id("ScalarFieldRGushort", "ScalarField", ScalarFieldRGmaker_ushort);

template <>
PersistentTypeID ScalarFieldRGT<char>::type_id("ScalarFieldRGchar", "ScalarField", ScalarFieldRGmaker_char);

template <>
PersistentTypeID ScalarFieldRGT<uchar>::type_id("ScalarFieldRGuchar", "ScalarField", ScalarFieldRGmaker_uchar);



template class ScalarFieldRGT<double>;
template class ScalarFieldRGT<float>;
template class ScalarFieldRGT<int>;
template class ScalarFieldRGT<short>;
template class ScalarFieldRGT<unsigned short>;
template class ScalarFieldRGT<char>;
template class ScalarFieldRGT<unsigned char>;

