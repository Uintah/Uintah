//////////////////////////////////////////////////////////////////////
// KDTree.h - Implements a template class for a KD-Tree.
// by David K. McAllister, Sep. 1999.

#ifndef _kdtree_h
#define _kdtree_h

// The class _Tp must support the following functions:
// friend inline bool lessX(const _Tp &a, const _Tp &b);
// friend inline bool lessY(const _Tp &a, const _Tp &b);
// friend inline bool lessZ(const _Tp &a, const _Tp &b);
// inline bool operator==(const _Tp &a) const;
// inline Vector &vector() const;
// These can point to the normal ones if no ties cannot occur.
// friend inline bool lessFX(const _Tp &a, const _Tp &b);
// friend inline bool lessFY(const _Tp &a, const _Tp &b);
// friend inline bool lessFZ(const _Tp &a, const _Tp &b);

#include <algorithm>

namespace Remote {
using namespace std;
#define MAX_BOX_SIZE 64

template<class _Tp>
class KDTree
{
  BBox Box;
  vector<_Tp> Items;
  _Tp Med;
  KDTree *left, *right;
  bool (*myless)(const _Tp &a, const _Tp &b);

public:
  inline KDTree() {left = right = NULL; myless = NULL;}

  inline KDTree(const _Tp *first, const _Tp *last)
  {
    left = right = NULL;
    myless = NULL;
    Items.assign(first, last);
    for(_Tp *I=Items.begin(); I!=Items.end(); I++)
      Box += I->Vert->V;

    // cerr << "Constructed KDTree with " << Items.size() << " items.\n";
  }

  inline ~KDTree()
  {
    if(left)
      delete left;
    if(right)
      delete right;
  }

  inline void Insert(const _Tp &It)
  {
    Box += It.vector();

    // fprintf(stderr, "I: 0x%08x ", long(this));
    // cerr << Items.size() << Box << " " << It.Vert->V << endl;

    if(left)
      {
	// Insert into the kids.
	if(myless(It, Med))
	  left->Insert(It);
	else
	  right->Insert(It);
	return;
      }
    
    Items.push_back(It);
    // fprintf(stderr, "A: 0x%08x\n", long(this));

    // Do I need to split the box?
    if(Items.size() > MAX_BOX_SIZE)
      {
	// Find which dimension.
	Vector d = Box.MaxV - Box.MinV;

	if(d.x > d.y) myless = d.x > d.z ? lessX : lessZ;
	else myless = d.y > d.z ? lessY : lessZ;

	// Split the box into two kids and find median.
	sort(Items.begin(), Items.end(), myless);

	_Tp *MedP = &Items[Items.size()/2];

	if(!myless(*(MedP-1), *MedP))
	  {
	    if(d.x > d.y) myless = d.x > d.z ? lessFX : lessFZ;
	    else myless = d.y > d.z ? lessFY : lessFZ;

	    sort(Items.begin(), Items.end(), myless);
	  }
	ASSERT1(myless(*(MedP-1), *MedP));
	Med = *MedP;

	// Puts the median in the right child.
	left = new KDTree(Items.begin(), MedP);
	right = new KDTree(MedP, Items.end());
	Items.clear();

	//	KDVertex Res;
      }
  }

  inline bool Find(const _Tp &It, _Tp &Res)
  {
    // fprintf(stderr, "F: 0x%08x ", long(this));
    // cerr << Items.size() << Box << " " << It.Vert->V << endl;

    if(left)
      {
	// Find into the kids.
	if(myless(It, Med))
	  return left->Find(It, Res);
	else
	  return right->Find(It, Res);
      }

    for(_Tp *I=Items.begin(); I!=Items.end(); I++)
      if(*I == It)
	{
	  Res = *I;
	  return true;
	}

    return false;
  }

private:
  // Finds the closest item in the same box as the query point.
  inline bool FNE(const _Tp &It, _Tp &Res, const double &DSqr = 0)
  {
    if(left)
      {
	// Find into the kids.
	if(myless(It, Med))
	  return left->FNE(It, Res);
	else
	  return right->FNE(It, Res);
      }

    for(_Tp *I=Items.begin(); I!=Items.end(); I++)
      if(VecEq(I->vector(), It.vector(), DSqr))
	{
	  Res = *I;
	  return true;
	}

    return false;
  }

  // Find a close enough item in any box.
  inline bool FNEB(const _Tp &It, _Tp &Res, const double &D = 0)
  {
    if(left)
      {
	if(left->Box.SphereIntersect(It.vector(), D))
	    if(left->FNEB(It, Res))
	      return true;
	if(right->Box.SphereIntersect(It.vector(), D))
	    if(right->FNEB(It, Res))
	      return true;
	return false;
      }
    
    double DSqr = Sqr(D);
    for(_Tp *I=Items.begin(); I!=Items.end(); I++)
      if(VecEq(I->vector(), It.vector(), DSqr))
	{
	  Res = *I;
	  return true;
	}

    return false;
  }

public:
  // Finds the closest item in the same box as the query point.
  inline bool FindNearEnough(const _Tp &It, _Tp &Res, const double &D = 0)
  {
    if(FNE(It, Res, Sqr(D)))
      return true;

    if(D > 0)
      return FNEB(It, Res, D);
    else
      return false;
  }

  void Dump()
  {
    cerr << Box << endl;
    
    if(left)
      {
	left->Dump();
	right->Dump();
      }
    else
      {
	cerr << "Count = " << Items.size() << endl;
	for(int i=0; i<Items.size(); i++)
	  {
	    Vector &V = Items[i].vector();
	    fprintf(stderr, "%d %0.20lf %0.20lf %0.20lf\n", i, V.x, V.y, V.z);
	    // cerr << i << " " << Items[i].vector() << endl;
	  }
      }
  }
};

} // End namespace Remote


#endif
