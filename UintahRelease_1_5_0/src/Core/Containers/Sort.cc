/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
    Sort.cc

*/

#include <Core/Containers/Sort.h>

#include <iostream>
#include <cstdlib>
using std::cerr;

namespace SCIRun {

const unsigned int KEY_SIZE=8;       // #bits for the Key
const int K = 1<<KEY_SIZE;  // size of psum table

const unsigned int BMASK=K-1; // bit mask for given key

const int NPASS=32/KEY_SIZE; // number of passes...

void SortObjs::DoRadixSort(Array1<unsigned int>& data, 
			   Array1<unsigned int>& index,
			   Array1<unsigned int>& nindex)
{
  keys.setsize(K);

  nindex.setsize(data.size());

//  nindex = index;

  unsigned int *indeces[2] = { &nindex[0], &index[0] };

  unsigned int *datap = &data[0]; // pointers faster?

  unsigned int back_shift=0;

  unsigned int cur_index=0;

  int i,j;

  for(i=0;i<NPASS;i++) {
    // toggle between index and nindex...
    keys.initialize(0); // reset this thing...

    for(j=0;j<data.size();j++) {
      unsigned int idex = (datap[j]>>back_shift)&BMASK;
      keys[idex]++;
    } // frequency of each value

    for(j=1;j<K;j++) {
      keys[j] += keys[j-1]; // prefix sum table...
    }

    unsigned int other_index = (cur_index+1)%2;

    for(j=data.size()-1;j<data.size();--j) { // will wrap...
      unsigned int ri = indeces[cur_index][j]; // real index to use...
      unsigned int ci = (datap[ri]>>back_shift)&BMASK;

      unsigned int thingy = keys[ci]-1;

      indeces[other_index][thingy] = ri;

      keys[ci]--;  // to deal with keys that have the same value...
    }
    
    back_shift += KEY_SIZE; // when sticking stuff in keys...

    if (cur_index) {
      cur_index = 0;
    } else {
      cur_index = 1;
    }
  }
}

void SortObjs::DoRadixSort(Array1<unsigned int>& data, Array1<unsigned int>& nindex)
{
  // build an identity index array...
  tmp_idex.setsize(data.size());

  nindex.setsize(data.size());

  for(int i=0;i<data.size();i++){
    nindex[i] = tmp_idex[i] = i;
  }

  DoRadixSort(data,tmp_idex,nindex);
}

const int K16 = 1<<16;  // size of psum table

const unsigned int BMASK16=K16-1; // bit mask for given key

void SortObjs::RSort16(Array1<unsigned int>& data, 
		      Array1<unsigned int>& sindex,
		      Array1<unsigned int>& dindex, int back_shift)
{
  keys.setsize(K16);

  unsigned int *src_ptr = &sindex[0];
  unsigned int *dst_ptr = &dindex[0];
  unsigned int *datap = &data[0];

  int j;
  

  keys.initialize(0);

  for(j=0;j<data.size();j++) {
    unsigned int idex = (datap[j]>>back_shift)&BMASK16;
    keys[idex]++;
  } // frequency of each value

  for(j=1;j<K16;j++) {
    keys[j] += keys[j-1]; // prefix sum table...
  }

  for(j=data.size()-1;j<data.size();--j) { // will wrap...
    unsigned int ri = src_ptr[j]; // real index to use...
    unsigned int ci = (datap[ri]>>back_shift)&BMASK16;
    
    unsigned int thingy = keys[ci]-1;
    
    dst_ptr[thingy] = ri;

    keys[ci]--;  // to deal with keys that have the same value...
  }
}

const int K12 = 1<<12;  // size of psum table

const unsigned int BMASK12=K12-1; // bit mask for given key

void SortObjs::RSort12(Array1<unsigned int>& data, 
		      Array1<unsigned int>& sindex,
		      Array1<unsigned int>& dindex, int back_shift)
{
  keys.setsize(K12);

  unsigned int *src_ptr = &sindex[0];
  unsigned int *dst_ptr = &dindex[0];
  unsigned int *datap = &data[0];
  
  int j;

  keys.initialize(0);

  for(j=0;j<data.size();j++) {
    unsigned int idex = (datap[j]>>back_shift)&BMASK12;
    keys[idex]++;
  } // frequency of each value

  for(j=1;j<K12;j++) {
    keys[j] += keys[j-1]; // prefix sum table...
  }

  for(j=data.size()-1;j<data.size();--j) { // will wrap...
    unsigned int ri = src_ptr[j]; // real index to use...
    unsigned int ci = (datap[ri]>>back_shift)&BMASK12;
    
    unsigned int thingy = keys[ci]-1;
    
    dst_ptr[thingy] = ri;

    keys[ci]--;  // to deal with keys that have the same value...
  }
}

const int K8 = 1<<8;  // size of psum table

const unsigned int BMASK8=K8-1; // bit mask for given key

void SortObjs::RSort8(Array1<unsigned int>& data, 
		      Array1<unsigned int>& sindex,
		      Array1<unsigned int>& dindex, int back_shift)
{
  keys.setsize(K8);

  unsigned int *src_ptr = &sindex[0];
  unsigned int *dst_ptr = &dindex[0];
  unsigned int *datap = &data[0];

  int j;
  

  keys.initialize(0);

  for(j=0;j<data.size();j++) {
    unsigned int idex = (datap[j]>>back_shift)&BMASK8;
    keys[idex]++;
  } // frequency of each value

  for(j=1;j<K8;j++) {
    keys[j] += keys[j-1]; // prefix sum table...
  }

  for(j=data.size()-1;j<data.size();--j) { // will wrap...
    unsigned int ri = src_ptr[j]; // real index to use...
    unsigned int ci = (datap[ri]>>back_shift)&BMASK8;
    
    unsigned int thingy = keys[ci]-1;
    
    dst_ptr[thingy] = ri;

    keys[ci]--;  // to deal with keys that have the same value...
  }
}

const int K4 = 1<<4;  // size of psum table

const unsigned int BMASK4=K4-1; // bit mask for given key

void SortObjs::RSort4(Array1<unsigned int>& data, 
		      Array1<unsigned int>& sindex,
		      Array1<unsigned int>& dindex, int back_shift)
{
  keys.setsize(K4);

  unsigned int *src_ptr = &sindex[0];
  unsigned int *dst_ptr = &dindex[0];
  unsigned int *datap = &data[0];
  int j;
  

  keys.initialize(0);

  for(j=0;j<data.size();j++) {
    unsigned int idex = (datap[j]>>back_shift)&BMASK4;
    keys[idex]++;
  } // frequency of each value

  for(j=1;j<K4;j++) {
    keys[j] += keys[j-1]; // prefix sum table...
  }

  for(j=data.size()-1;j<data.size();--j) { // will wrap...
    unsigned int ri = src_ptr[j]; // real index to use...
    unsigned int ci = (datap[ri]>>back_shift)&BMASK4;
    
    unsigned int thingy = keys[ci]-1;
    
    dst_ptr[thingy] = ri;

    keys[ci]--;  // to deal with keys that have the same value...
  }
}

/*
 * Bit selecion is kind of tricky - for a given number
 * of passes, you want to use the keys that have the minimum
 * max value - for example 12 + 8 is better than 16 + 4
 *
 */

void SortObjs::DoSmartRadixSort(Array1<unsigned int>& data, 
				Array1<unsigned int> &nindex,
				int nbits)
{
  tmp_idex.setsize(data.size());
  nindex.setsize(data.size());

  // do minimal number of passes with smallest number of 
  // bits - you can overflow - just not past
  int n16=0,n12=0,n8=0,n4=0;
  int tot_bits=0;
  //  int rnbits = nbits;
  int npass=0;
#if 0
  while((nbits >= 16) || ((tot_bits <= (32-16)) && (nbits > 12))) {
    tot_bits += 16;
    n16++;
    nbits -= 16;
    npass++;
  }

#endif
  while((nbits >= 12) || ((tot_bits <= (32-12)) && (nbits > 8))) {
    tot_bits += 12;
    n12++;
    nbits -= 12;
    npass++;
  }

  while((nbits >= 8) || ((tot_bits <= (32-8)) && (nbits > 4))) {
    tot_bits += 8;
    n8++;
    nbits -= 8;
    npass++;
  }
  
  while((nbits >= 4) || ((tot_bits < (32-4)) && (nbits > 0))) {
    tot_bits += 4;
    n4++;
    nbits -= 4;
    npass++;
  }
  
  // now do some adjustments...

  while (n16 && n4) {
    n12++;n8++;
    --n16;--n4;
  }

  while(n16 && n8) {
    n12 += 2;
    --n16;--n8;
  }

  while(n12 && n4) {
    n8 += 2;
    --n12; --n4;
  }

  if (nbits > 0) {
    cerr << "Woah - big problem with optimal bit allocation...\n";
  }

//  cerr << nbits << " " << n16 << " " << n12 << " " << n8 << " " << n4 << " ";

  int use_nindex = (npass&1);

  int i;

  if (use_nindex) {
    for(i=0;i<data.size();i++) {
      tmp_idex[i] = i;
    }
  } else {
    for(i=0;i<data.size();i++) {
      nindex[i] = i;
    }
  }

  int bshift=0;


  while(n16--) {
    if (use_nindex) {
      RSort16(data,tmp_idex,nindex,bshift);
      use_nindex=0;
    } else {
      RSort16(data,nindex,tmp_idex,bshift);
      use_nindex=1;
    }
    
    bshift += 16;
  }
  while(n12--) {
    if (use_nindex) {
      RSort12(data,tmp_idex,nindex,bshift);
      use_nindex=0;
    } else {
      RSort12(data,nindex,tmp_idex,bshift);
      use_nindex=1;
    }
    
    bshift += 12;
  }
  while(n8--) {
    if (use_nindex) {
      RSort8(data,tmp_idex,nindex,bshift);
      use_nindex=0;
    } else {
      RSort8(data,nindex,tmp_idex,bshift);
      use_nindex=1;
    }
    
    bshift += 8;
  }
  while(n4--) {
    if (use_nindex) {
      RSort4(data,tmp_idex,nindex,bshift);
      use_nindex=0;
    } else {
      RSort4(data,nindex,tmp_idex,bshift);
      use_nindex=1;
    }
    
    bshift += 4;
  }
}

struct QSortHelperF {
  float *data;  // this is kind of memmory wastefull...
  int id;
};

int SimpCompF(const void* e1, const void* e2)
{
  QSortHelperF *a = (QSortHelperF*)e1;
  QSortHelperF *b = (QSortHelperF*)e2;

  if (a->data[a->id] >
      a->data[b->id])
    return 1;
  if (a->data[a->id] <
      a->data[b->id])
    return -1;

  return 0; // they are equal...  
}


struct QSortHelperI {
  unsigned int *data;  // this is kind of memmory wastefull...
  int id;
};

int SimpCompI(const void* e1, const void* e2)
{
  QSortHelperI *a = (QSortHelperI*)e1;
  QSortHelperI *b = (QSortHelperI*)e2;

  if (a->data[a->id] >
      a->data[b->id])
    return 1;
  if (a->data[a->id] <
      a->data[b->id])
    return -1;

  return 0; // they are equal...  
}

void SortObjs::DoQSort(Array1<float>& data, Array1<unsigned int>& index)
{
  Array1<QSortHelperF> helpers;
  int i;

  helpers.resize(data.size());

  for(i=0;i<helpers.size();i++) {
    helpers[i].data = &data[0];
    helpers[i].id = i;
  }

  qsort(&helpers[0],helpers.size(),sizeof(QSortHelperF),SimpCompF);

  // now just dump the data back...

  index.setsize(data.size());
  for(i=0;i<helpers.size();i++) {
    index[i] = helpers[i].id;
  }
}



void SortObjs::DoQSort(Array1<unsigned int>& data, Array1<unsigned int>& index)
{
  Array1<QSortHelperI> helpers;
  int i;

  helpers.resize(data.size());

  for(i=0;i<helpers.size();i++) {
    helpers[i].data = &data[0];
    helpers[i].id = i;
  }

  qsort(&helpers[0],helpers.size(),sizeof(QSortHelperI),SimpCompI);

  // now just dump the data back...

  index.setsize(data.size());
  for(i=0;i<helpers.size();i++) {
    index[i] = helpers[i].id;
  }
}

// test the two sorting methods...

#if 0

const int nvals=4000000;
unsigned int NUM_SHIFT=19;
unsigned int SCALE_FAC=1<<NUM_SHIFT; // some really big number...
void main(int argc, char* argv)
{
  Array1<float> vals(nvals);
  Array1<unsigned int>   ivals(nvals);
  int i,j;

  Array1<unsigned int> index(nvals);

  cerr << KEY_SIZE << " " << K12 << " " << BMASK12 << " " << NPASS << endl;

  for(i=0;i<nvals;i++) {
    vals[i] = drand48();  // make it a random number
  }

  WallClockTimer mytime;
  SortObjs sorter;

  //double minv=0.0;
  //double mfac=1.0;

  for (NUM_SHIFT=2;NUM_SHIFT<32;NUM_SHIFT++) {
    SCALE_FAC=1<<NUM_SHIFT;
  
    
    // first quantize them...

    for(i=0;i<nvals;i++) {
      ivals[i] = vals[i]*SCALE_FAC;
    }

    for (j=0;j<10;j++) {

      mytime.start();

      // then sort them...

      //  sorter.DoRadixSort(ivals,index);

      sorter.DoSmartRadixSort(ivals,index,NUM_SHIFT+1);
      //  sorter.DoRadixSort(ivals,index);

      mytime.stop();
    }
    cerr << NUM_SHIFT+1 << " Radix Sort: " << mytime.time() << endl;
  
    mytime.clear(); // clear the timer...
  }

  for(i=1;i<nvals;i++) {
    if ((vals[index[i-1]] > vals[index[i]])) {
      if (ivals[index[i-1]] > ivals[index[i]]) {
	cerr << "Error: " << vals[index[i-1]] << ":" << vals[index[i]] <<  endl;
	cerr << " " << ivals[index[i-1]] << " : " << ivals[index[i]] << endl;
      }
    }
  }


  mytime.start();

  sorter.DoQSort(ivals,index);

  mytime.stop();

  cerr << "Quick Sort: " << mytime.time() << endl;
#if 0
  for(i=1;i<nvals;i++) {
    if (vals[index[i-1]] > vals[index[i]])
      cerr << "Error: " << vals[index[i]] << endl;
  }
#else
  for(i=1;i<nvals;i++) {
    if ((vals[index[i-1]] > vals[index[i]])) {
      if (ivals[index[i-1]] > ivals[index[i]]) {
	cerr << "Error: " << vals[index[i-1]] << ":" << vals[index[i]] <<  endl;
	cerr << " " << ivals[index[i-1]] << " : " << ivals[index[i]] << endl;
      }
    }
  }

#endif
}

#endif

} // End namespace SCIRun


