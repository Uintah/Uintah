
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
  : ScalarFieldRGBase("ushort")
{
}

template <>
ScalarFieldRGT<char>::ScalarFieldRGT()
  : ScalarFieldRGBase("char")
{
}

template <>
ScalarFieldRGT<unsigned char>::ScalarFieldRGT()
  : ScalarFieldRGBase("uchar")
{
}


template <>
PersistentTypeID ScalarFieldRGT<double>::type_id("ScalarFieldRGdouble", "ScalarField", maker);

template <>
PersistentTypeID ScalarFieldRGT<float>::type_id("ScalarFieldRGfloat", "ScalarField", maker);

template <>
PersistentTypeID ScalarFieldRGT<int>::type_id("ScalarFieldRGint", "ScalarField", maker);

template <>
PersistentTypeID ScalarFieldRGT<short>::type_id("ScalarFieldRGshort", "ScalarField", maker);

template <>
PersistentTypeID ScalarFieldRGT<ushort>::type_id("ScalarFieldRGushort", "ScalarField", maker);

template <>
PersistentTypeID ScalarFieldRGT<char>::type_id("ScalarFieldRGchar", "ScalarField", maker);

template <>
PersistentTypeID ScalarFieldRGT<uchar>::type_id("ScalarFieldRGuchar", "ScalarField", maker);



template class ScalarFieldRGT<double>;
template class ScalarFieldRGT<float>;
template class ScalarFieldRGT<int>;
template class ScalarFieldRGT<short>;
template class ScalarFieldRGT<unsigned short>;
template class ScalarFieldRGT<char>;
template class ScalarFieldRGT<unsigned char>;

