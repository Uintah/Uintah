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



/*
 *  MxNArrayRep.cc 
 *
 *  Written by:
 *   Kostadin Damevski 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002 
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <iostream>
using namespace SCIRun;
using namespace std;

int SCIRun::gcd(int a, int b, int &x, int &y) {
  std::vector<int> q;
  int ret;
  if (a > b) {
    while ((a % b) != 0) {
      q.push_back((int)a/b);
      int t = a;
      a = b;
      b = t % a;      
    }
    ret = b;
  }
  else {
    while ((b % a) != 0) {
      q.push_back((int)b/a);
      int t = b;
      b = a;
      a = t % b;      
    }
    ret = a;
  }

  int size = q.size();
  int * s = new int[size+2];
  s[0] = 1; s[1]=0;
  for(int i=2; i < (size+2); i++) {
    s[i] = s[i-2] - (q[i-2]*s[i-1]);
  }

  int * t = new int[size+2];
  t[0] = 0; t[1]=1;
  for(int i=2; i < (size+2); i++) {
    t[i] = t[i-2] - (q[i-2]*t[i-1]);
  }  

  if (a > b) {
    x = s[size+1];
    y = t[size+1];
  } else {
    x = t[size+1];
    y = s[size+1];
  }

  delete[] s;
  delete[] t;

  return ret;
}

int SCIRun::intersectSlice(int f1, int s1, int f2, int s2)
{
  int I;
  int m,n;
  int _gcd = gcd(s1,s2,m,n);
  int _lcm = lcm(s1,s2);

  I = f1 + (- ((s1*m*(f1-f2))/_gcd));
  return (I + (_lcm * (int)ceil((double)(std::max(f1,f2) - I) / (double)_lcm)));
} 

int SCIRun::gcd(int m,int n) {
  if(n == 0)
    return m;
  else 
    return (SCIRun::gcd(n, m % n));
}

int SCIRun::lcm(int m,int n) {
  return ((m*n) / gcd(m,n));
}

MxNArrayRep::MxNArrayRep(int dimno, Index* dimarr[], Reference* remote_ref) 
  : mydimno(dimno)
{
  //for(int i=0; i < mydimno; i++){
  for(int i=mydimno-1; i >= 0; i--){
    mydimarr.push_back( *(dimarr[i]) );
  }
  if (remote_ref != NULL)  this->remoteRef = remote_ref;
  received = false;
}

MxNArrayRep::MxNArrayRep(SSIDL::array2<int>& arr, Reference* remote_ref) 
{
  mydimno = arr.size2();
  //for(int i=0; i < mydimno; i++){
  for(int i=mydimno-1; i >= 0; i--){
    mydimarr.push_back(Index(arr[0][i],arr[1][i],arr[2][i]) );
  }
  if (remote_ref != NULL)  this->remoteRef = remote_ref;
  received = false;
}

MxNArrayRep::~MxNArrayRep()
{
}

SSIDL::array2<int> MxNArrayRep::getArray()
{
  SSIDL::array2<int> dist(3,mydimno);
  for(int i=0; i<mydimno; i++) {
    dist[0][i] = mydimarr[i].myfirst;
    dist[1][i] = mydimarr[i].mylast;
    dist[2][i] = mydimarr[i].mystride;    
  }
  return dist;
}

unsigned int MxNArrayRep::getDimNum()
{
  return mydimno;
}

unsigned int MxNArrayRep::getFirst(int dimno)
{
  if (dimno <= mydimno)
    return mydimarr[dimno-1].myfirst;
  else
    return 0;
}

unsigned int MxNArrayRep::getLast(int dimno)
{
  if (dimno <= mydimno)
    return mydimarr[dimno-1].mylast;
  else
    return 0;
}

unsigned int MxNArrayRep::getStride(int dimno)
{
  if (dimno <= mydimno)
    return mydimarr[dimno-1].mystride;
  else
    return 1;
}

unsigned int MxNArrayRep::getLocalStride(int dimno)
{
  if (dimno <= mydimno) {
    return mydimarr[dimno-1].localStride;
  }
  else
    return 1;
}

unsigned int MxNArrayRep::getSize(int dimno)
{
  if (dimno <= mydimno) {
    int fst = mydimarr[dimno-1].myfirst;
    int lst = mydimarr[dimno-1].mylast;
    int str = mydimarr[dimno-1].mystride;
    int localStride = mydimarr[dimno-1].localStride;
    return ( ((int) ceil((float)(lst - fst) / (float)str) ) * localStride );
  }
  else
    return 0;
}

Reference* MxNArrayRep::getReference()
{
  return remoteRef;
}

void MxNArrayRep::setRank(int rank)
{
  d_rank = rank;
}

int MxNArrayRep::getRank()
{
  return d_rank;
}

bool MxNArrayRep::isIntersect(MxNArrayRep* arep)
{
  bool intersect = true;

  if (arep != NULL) {
    //Call Intersect()
    MxNArrayRep* result = this->Intersect(arep);
    //Loop through the result to see if there is an intersection
    for(int j=1; j <= mydimno ; j++) {
      if (result->getFirst(j) > result->getLast(j)) {
	intersect = false;
	break;
      }
      if (result->getStride(j) < 1) {
	intersect = false;
	break;
      }
    }

    return intersect;
  }
  return false;
}

Index* MxNArrayRep::Intersect(MxNArrayRep* arep, int dimno)
{
  Index* intersectionRep = new Index(0,0,0);
  if (arep != NULL) {
    if (dimno <= mydimno) {

      int myfst,mylst,mystr;  //The data from this representation
      int fst,lst,str;        //The data from the intersecting representation

      //Get the representations to be interested:
      myfst = mydimarr[dimno-1].myfirst;
      mylst = mydimarr[dimno-1].mylast;
      mystr = mydimarr[dimno-1].mystride;
      fst = arep->getFirst(dimno);
      lst = arep->getLast(dimno);
      str = arep->getStride(dimno);
      
      //Intersecting to find the first,last and stride of the intersection:

      //Calculate lcm of the strides:
      int lcm_str = lcm(str,mystr);
      int d_gcd = gcd(str,mystr);
      intersectionRep->mystride = lcm_str;

      if ((fst % d_gcd) == (myfst % d_gcd)) {
	intersectionRep->myfirst = intersectSlice(fst,str,myfst,mystr);
      } 
      else {
	//No Intersection
	intersectionRep->myfirst = 0;
	intersectionRep->mylast = 0;
	intersectionRep->mystride = 0;

	//intersectionRep->print(std::cerr);
	return intersectionRep;
      }

      //Find the last
      int min_lst = std::min(mylst,lst) - 1;
      int dif = min_lst - intersectionRep->myfirst;
      if (dif > 0) {
	intersectionRep->mylast = min_lst;
      }
      else {
	//No Intersection, leave results as they will signify that also
	intersectionRep->mylast = min_lst;
      }
      //std::cerr << "INTERSECTION: ";
      //intersectionRep->print(std::cerr);
    }
  }
  return intersectionRep;
}

MxNArrayRep* MxNArrayRep::Intersect(MxNArrayRep* arep)
{
  Index* intersectionArr[mydimno];
  for(int i=1; i<=mydimno; i++) {
    intersectionArr[i-1] = Intersect(arep, i);
  }
  return (new MxNArrayRep(mydimno,intersectionArr,arep->getReference()));
}

void MxNArrayRep::print(std::ostream& dbg)
{
  for(int i=0; i < mydimno ; i++) {
    mydimarr[i].print(dbg);
  }  
}


//******************  Index ********************************************

Index::Index(unsigned int first, unsigned int last, unsigned int stride, int localStride)
  : mystride(stride), localStride(localStride)
{
  if (first > last) {
    myfirst = last;
    mylast = first;
  }
  else {
    myfirst = first;
    mylast = last;
  }
}

void Index::print(std::ostream& dbg)
{
  dbg << "["<<myfirst << ", " << mylast << ") stride=" << mystride << "\n";
}    

    











