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


#include <Tester/TestTable.h>
#include <Containers/String.h>
#include <Containers/Array1.h>
#include <Containers/Array2.h>
#include <Containers/Array3.h>
#include <Containers/HashTable.h>
#include <Containers/FastHashTable.h>
#include <Containers/BitArray1.h>

namespace SCIRun {


TestTable test_table[] = {
    {"Array1", Array1<float>::test_rigorous, 0},
    {"Array2", Array2<int>::test_rigorous, 0},
    {"Array3", Array3<int>::test_rigorous, 0},
//  {"HashTable", HashTable<char*, int>::test_rigorous, 0},
    {"FastHashTable", FastHashTable<int>::test_rigorous, 0},
    {0,0,0}

};

} // End namespace SCIRun


