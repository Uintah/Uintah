/*!
    Test the following:
    	MxNArrayRep Creation and Usage
*/

#include <iostream>
#include <assert.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>
using namespace SCIRun;
using namespace std;

int main()
{

  MxNArrayRep* mxnarep1;
  MxNArrayRep* mxnarep2;
  MxNArrayRep* mxnarep3;
  
  //Create an empty ArrayRep and test some trivial things about it
  mxnarep1 = new MxNArrayRep(0,NULL);
  assert(mxnarep1->getDimNum() == 0);
  assert(mxnarep1->getFirst(2) == 0);

  //Create another ArrayRep
  Index** dr = new Index* [1];
  dr[0] = new Index(33,38,3);
  mxnarep2 = new MxNArrayRep(1,dr);
  assert(mxnarep2->getDimNum() == 1);
  assert(mxnarep2->getFirst(1) == 33);
  assert(mxnarep2->getLast(1) == 38);
  assert(mxnarep2->getStride(1) == 3);
  assert(mxnarep2->getSize(1) == 2);


  //Create an ArrayRep from an array 
  SSIDL::array2<int> arr1(3,2);
  arr1[0][0] = 29; arr1[1][0] = 101; arr1[2][0] = 4;
  arr1[0][1] = 23; arr1[1][1] = 107; arr1[2][1] = 5;
  mxnarep3 = new MxNArrayRep(arr1);
  assert(mxnarep3->getDimNum() == 2);
  assert(mxnarep3->getFirst(2) == 23);
  assert(mxnarep3->getLast(1) == 101);
  assert(mxnarep3->getStride(2) == 5);
  assert(mxnarep3->getSize(2) == 17);

  //Linear Diophantine solution tests
  int t,x,y;
  t = gcd(31,7,x,y);
  assert(y == 9); 
  t = gcd(7,12,x,y);
  assert(x == -5); 

  //Intersection tests
  assert(mxnarep3->isIntersect(mxnarep2) == true);
  Index* nsect1 = mxnarep3->Intersect(mxnarep2,1);
  assert(nsect1->myfirst == 33); 
  assert(nsect1->mylast == 37);
  delete nsect1;

  delete mxnarep2;
  Index** index1 = new Index* [2];
  index1[0] = new Index(0,44,1);
  index1[1] = new Index(0,98,2);
  mxnarep2 = new MxNArrayRep(2,index1);  
  MxNArrayRep* xsect = mxnarep2->Intersect(mxnarep3);
  assert(xsect->getFirst(1) == 29);
  assert(xsect->getLast(1) == 43);
  assert(xsect->getFirst(2) == 28);
  assert(xsect->getLast(2) == 97);
  delete xsect;
  
  delete mxnarep1;
  Index** index2 = new Index* [2];
  index2[0] = new Index(46,99,4);
  index2[1] = new Index(33,65,1);
  mxnarep1 = new MxNArrayRep(2,index2);  
  MxNArrayRep* xsect1 = mxnarep1->Intersect(mxnarep2);
  assert(xsect1->getFirst(1) == 46);
  assert(xsect1->getLast(1) == 43);
  assert(xsect1->getStride(1) == 4);
  assert(xsect1->getFirst(2) == 34);
  assert(xsect1->getLast(2) == 64);
  assert(xsect1->getStride(2) == 2);
  delete xsect1; 

  MxNArrayRep* xsect2 = mxnarep1->Intersect(mxnarep3);
  assert(xsect2->getFirst(1) == 0);
  assert(xsect2->getLast(1) == 0);
  assert(xsect2->getStride(1) == 0);
  assert(xsect2->getFirst(2) == 33);
  assert(xsect2->getLast(2) == 64);
  assert(xsect2->getStride(2) == 5);
  delete xsect1;

  delete mxnarep1;
  delete mxnarep2;
  delete mxnarep3;
}


