
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
ScalarFieldRGT<double>::ScalarFieldRGT(int x, int y, int z)
  : ScalarFieldRGBase("double", x, y, z),
    grid(x, y, z)
{
}

template <>
ScalarFieldRGT<float>::ScalarFieldRGT(int x, int y, int z)
  : ScalarFieldRGBase("float", x, y, z),
    grid(x, y, z)
{
}

template <>
ScalarFieldRGT<int>::ScalarFieldRGT(int x, int y, int z)
  : ScalarFieldRGBase("int", x, y, z),
    grid(x, y, z)
{
}

template <>
ScalarFieldRGT<short>::ScalarFieldRGT(int x, int y, int z)
  : ScalarFieldRGBase("short", x, y, z),
    grid(x, y, z)
{
}

template <>
ScalarFieldRGT<unsigned short>::ScalarFieldRGT(int x, int y, int z)
  : ScalarFieldRGBase("ushort", x, y, z),
    grid(x, y, z)
{
}

template <>
ScalarFieldRGT<char>::ScalarFieldRGT(int x, int y, int z)
  : ScalarFieldRGBase("char", x, y, z),
    grid(x, y, z)
{
}

template <>
ScalarFieldRGT<unsigned char>::ScalarFieldRGT(int x, int y, int z)
  : ScalarFieldRGBase("uchar", x, y, z),
    grid(x, y, z)
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

