/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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


