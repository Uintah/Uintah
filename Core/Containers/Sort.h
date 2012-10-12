/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 * This provides several functions that will sort data
 * for you.  One uses the qsort routine, the other uses
 * a radix sort.
 */

#ifndef SCI_Containers_SORT_H_
#define SCI_Containers_SORT_H_ 1

#include <Core/Containers/Array1.h>

namespace SCIRun {

class SortObjs {
public:
  SortObjs() {};

  // ndata is a "scratch" array
  void DoRadixSort(Array1<unsigned int>& data, Array1<unsigned int>& index, 
		   Array1<unsigned int>& nindex);

  // below just builds identity index...
  void DoRadixSort(Array1<unsigned int>& data, Array1<unsigned int>& nindex);

  // below intelligently chooses keys...

  void DoSmartRadixSort(Array1<unsigned int>& data, 
			Array1<unsigned int> &nindex,
			int nbits); // number of bits used in encoding

  void DoQSort(Array1<float>& data, Array1<unsigned int>& nindex);

  void DoQSort(Array1<unsigned int>& data, Array1<unsigned int>& nindex);

//  void DoQSort(Array1<int>& data, Array1<int>& index, Array1<int>& nindex);
protected:

  // these functions all just sort for a fixed "key" size

  void RSort16(Array1<unsigned int>& data, Array1<unsigned int>& sindex,
	       Array1<unsigned int>& dindex, int bshift);
  void RSort12(Array1<unsigned int>& data, Array1<unsigned int>& sindex,
	       Array1<unsigned int>& dindex, int bshift);
  void RSort8(Array1<unsigned int>& data, Array1<unsigned int>& sindex,
	      Array1<unsigned int>& dindex, int bshift);
  void RSort4(Array1<unsigned int>& data, Array1<unsigned int>& sindex,
	      Array1<unsigned int>& dindex, int bshift);
/* later down the road you might want this... */
/*
  void RSort2(Array1<unsigned int>& data, Array1<unsigned int>& sindex,
	      Array1<unsigned int>& dindex, int bshift);
*/

  Array1<unsigned int> keys; // helper data...

  Array1<unsigned int> tmp_idex; // temporary index...
};

} // End namespace SCIRun


#endif
