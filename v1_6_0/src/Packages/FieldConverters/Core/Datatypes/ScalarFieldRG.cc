/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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


#include <FieldConverters/Core/Datatypes/ScalarFieldRG.h>

using namespace FieldConverters;
using namespace SCIRun;

template <>
ScalarFieldRGT<double>::ScalarFieldRGT(int x, int y, int z)
  : ScalarFieldRGBase(Double, x, y, z),
    grid(x, y, z)
{
}

template <>
ScalarFieldRGT<int>::ScalarFieldRGT(int x, int y, int z)
  : ScalarFieldRGBase(Int, x, y, z),
    grid(x, y, z)
{
}

template <>
ScalarFieldRGT<short>::ScalarFieldRGT(int x, int y, int z)
  : ScalarFieldRGBase(Short, x, y, z),
    grid(x, y, z)
{
}

template <>
ScalarFieldRGT<char>::ScalarFieldRGT(int x, int y, int z)
  : ScalarFieldRGBase(Char, x, y, z),
    grid(x, y, z)
{
}

template <>
PersistentTypeID ScalarFieldRGT<double>::type_id("ScalarFieldRG", "ScalarField", maker);

template <>
PersistentTypeID ScalarFieldRGT<int>::type_id("ScalarFieldRGint", "ScalarField", maker);

template <>
PersistentTypeID ScalarFieldRGT<short>::type_id("ScalarFieldRGshort", "ScalarField", maker);

template <>
PersistentTypeID ScalarFieldRGT<char>::type_id("ScalarFieldRGchar", "ScalarField", maker);



template class ScalarFieldRGT<double>;
template class ScalarFieldRGT<int>;
template class ScalarFieldRGT<short>;
template class ScalarFieldRGT<char>;
