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
 * This provides several functions that will sort data
 * for you.  One uses the qsort routine, the other uses
 * a radix sort.
 */

#ifndef SCI_Containers_SORT_H_
#define SCI_Containers_SORT_H_ 1

#include <Core/share/share.h>

#include <Core/Containers/Array1.h>

namespace SCIRun {

class SCICORESHARE SortObjs {
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
