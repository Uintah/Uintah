//  DiscreteAttrib.h
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   August 2000
//  Copyright (C) 2000 SCI Institute
//  Attribute containing a finite number of discrete values.

#ifndef SCI_project_DiscreteAttrib_h
#define SCI_project_DiscreteAttrib_h 1

#include <vector>

#include <Core/Datatypes/Attrib.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/DebugStream.h>
#include <memory.h>
#include <iostream>
using namespace std;
#ifdef __sgi
#include <limits>
#endif

namespace SCIRun {

template <class T> class AttribFunctor
{
public:
  virtual void operator() (T &) {}
};

template <class T> class DiscreteAttrib : public Attrib
{
public:
  DiscreteAttrib();
  DiscreteAttrib(int x);
  DiscreteAttrib(int x, int y);
  DiscreteAttrib(int x, int y, int z);
  
  DiscreteAttrib(const DiscreteAttrib& copy);

  virtual ~DiscreteAttrib();

  virtual void get1(T &result, int x);
  virtual void get2(T &result, int x, int y);
  virtual void get3(T &result, int x, int y, int z);

  virtual T &get1(int x);
  virtual T &get2(int x, int y);
  virtual T &get3(int x, int y, int z);

  T &fget1(int x);
  T &fget2(int x, int y);
  T &fget3(int x, int y, int z);
  
  virtual void set1(int x, const T &val);
  virtual void set2(int x, int y, const T &val);
  virtual void set3(int x, int y, int z, const T &val);

  void fset1(int x, const T &val);
  void fset2(int x, int y, const T &val);
  void fset3(int x, int y, int z, const T &val);
  
  // Implement begin()
  // Implement end()
  
  // Resize the attribute to the specified dimensions
  virtual void resize(int, int, int);
  virtual void resize(int, int);
  virtual void resize(int);
  
  int size() const;
  
  //////////
  // Attribute validity check
  virtual bool isValid(int) const;
  virtual bool isValid(int, int) const;
  virtual bool isValid(int, int, int) const;
  virtual int  getNumOfValid() const;
  
  virtual void setValidBit(int, bool);
  virtual void setValidBit(int, int, bool);
  virtual void setValidBit(int, int, int, bool);
  virtual void setValidBit(const IntVector& , const IntVector&, bool);
    
  virtual bool getValid1(int x, T&);
  virtual bool getValid2(int x, int y, T&);
  virtual bool getValid3(int x, int y, int z, T&);
  const unsigned int* getValidBits(int&);
  void copyValidBits(const unsigned int*, int);
  
  virtual string getInfo();

  //////////
  // Persistent representation...
  virtual void io(Piostream &);
  static PersistentTypeID type_id;
  static string typeName();
  static Persistent* maker();

  virtual int iterate(AttribFunctor<T> &func);

  virtual int xsize() const { return d_nx; }
  virtual int ysize() const { return d_ny; }
  virtual int zsize() const { return d_nz; }
  virtual int dimension() const { return d_dim; }
  virtual void initialize(const T&);
protected:
  
  // GROUP: Protected member functions
  //////////
  // -- returns size of validBits array in number of int's
  inline int      resizeValidBits(int);
  //////////
  // -- returns true if bit of supplied number is valid one
  inline bool     testValidBit(int) const;

  //////////
  // set bits to specified value in the supplied range
  inline void     validSet(int, int, bool);
  inline void     validSet(int, bool);

  // GROUP: protected data
  //////////
  // 

#ifdef __sgi
  static const int nbits = sizeof(unsigned int)*numeric_limits<unsigned char>::digits;
#else
  static const int nbits = sizeof(unsigned int)*8;
#endif

  static const unsigned int maxUI;

  /////////
  // Sizes and dimensionality
  int d_nx, d_ny, d_nz;
  int d_dim;

  unsigned int*   d_pValidBits;
  int             d_nalloc;
 
private:
  T d_defval;
  static DebugStream dbg;
};

//////////
// Static members for PIO support
template <class T> 
DebugStream DiscreteAttrib<T>::dbg("DiscreteAttrib", true);
 

template <class T>
string DiscreteAttrib<T>::typeName(){
  static string typeName = string("DiscreteAttrib<") + findTypeName((T*)0)+">";
  return typeName;
}

template <class T> Persistent*
DiscreteAttrib<T>::maker(){
  return new DiscreteAttrib<T>();
}

#ifdef __sgi

template <class T> 
const unsigned int DiscreteAttrib<T>::maxUI = numeric_limits<unsigned int>::max();

#else

template <class T> 
const unsigned int DiscreteAttrib<T>::maxUI = 0xFFFF;

#endif


template <class T> 
PersistentTypeID DiscreteAttrib<T>::type_id(DiscreteAttrib<T>::typeName(), 
					    "Attrib", 
					    DiscreteAttrib<T>::maker);

#define DISCRETEATTRIB_VERSION 1

template <class T> void
DiscreteAttrib<T>::io(Piostream& stream)
{
  stream.begin_class(typeName().c_str(), DISCRETEATTRIB_VERSION);
  
  // -- base class PIO
  Attrib::io(stream);
  Pio(stream, d_nx);
  Pio(stream, d_ny);
  Pio(stream, d_nz);
  Pio(stream, d_dim);
  Pio(stream, d_defval);
  Pio(stream, d_nalloc);
  
  if(stream.reading()){
    delete[] d_pValidBits;        // -- in case we're reading different file into the object
    d_pValidBits = new unsigned int[d_nalloc];
  }
  
  for (int i=0; i<d_nalloc; i++)
    Pio(stream, d_pValidBits[i]);
  
  stream.end_class();
}

//////////
// Constructors/Destructor
template <class T>
DiscreteAttrib<T>::DiscreteAttrib() :
  Attrib(), d_nx(0), d_ny(0), d_nz(0), 
  d_dim(0),
  d_pValidBits(0)
{
}

template <class T>
DiscreteAttrib<T>::DiscreteAttrib(int ix) :
  Attrib(), d_nx(ix), d_ny(0), d_nz(0), d_dim(1)
{
  d_pValidBits = 0;
  resizeValidBits(ix);
}

template <class T>
DiscreteAttrib<T>::DiscreteAttrib(int ix, int iy) :
  Attrib(), d_nx(ix), d_ny(iy), d_nz(0), d_dim(2)
{
  d_pValidBits = 0;
  resizeValidBits(ix*iy);
}

template <class T>
DiscreteAttrib<T>::DiscreteAttrib(int ix, int iy, int iz) :
  Attrib(), d_nx(ix), d_ny(iy), d_nz(iz), d_dim(3)
{
  d_pValidBits = 0;
  resizeValidBits(ix*iy*iz);
}

template <class T>
DiscreteAttrib<T>::DiscreteAttrib(const DiscreteAttrib& copy) :
  Attrib(copy),
  d_nx(copy.d_nx), d_ny(copy.d_ny), d_nz(copy.d_nz),
  d_dim(copy.d_dim)
{
  d_pValidBits = 0;
  resizeValidBits(d_nx*d_ny*d_nz);
  memcpy(d_pValidBits, copy.d_pValidBits, d_nalloc*sizeof(unsigned int));
}

template <class T>
DiscreteAttrib<T>::~DiscreteAttrib()
{
  delete[] d_pValidBits;
}


template <class T> T &
DiscreteAttrib<T>::fget1(int)
{
  return d_defval;
}


template <class T> T &
DiscreteAttrib<T>::fget2(int, int)
{
  return d_defval;
}


template <class T> T &
DiscreteAttrib<T>::fget3(int, int, int)
{
  return d_defval;
}


// Copy wrappers, no allocation of result.
template <class T> void
DiscreteAttrib<T>::get1(T &result, int ix)
{
  result = fget1(ix);
}

template <class T> void
DiscreteAttrib<T>::get2(T &result, int ix, int iy)
{
  result = fget2(ix, iy);
}

template <class T> void
DiscreteAttrib<T>::get3(T &result, int ix, int iy, int iz)
{
  result = fget3(ix, iy, iz);
}


// Virtual wrappers for inline functions.
template <class T> T &
DiscreteAttrib<T>::get1(int ix)
{
  return fget1(ix);
}

template <class T> T &
DiscreteAttrib<T>::get2(int ix, int iy)
{
  return fget2(ix, iy);
}

template <class T> T &
DiscreteAttrib<T>::get3(int ix, int iy, int iz)
{
  return fget3(ix, iy, iz);
}




template <class T> void
DiscreteAttrib<T>::fset1(int, const T &val)
{
  d_defval = val;
}


template <class T> void
DiscreteAttrib<T>::fset2(int, int, const T &val)
{
  d_defval = val;
}


template <class T> void
DiscreteAttrib<T>::fset3(int, int, int, const T &val)
{
  d_defval = val;
}


// Generic setters for Discrete type
template <class T> void
DiscreteAttrib<T>::set1(int x, const T &val)
{
  fset1(x, val);
}

template <class T> void
DiscreteAttrib<T>::set2(int x, int y, const T &val)
{
  fset2(x, y, val);
}

template <class T> void
DiscreteAttrib<T>::set3(int x, int y, int z, const T &val)
{
  fset3(x, y, z, val);
}


// template <class T> bool DiscreteAttrib<T>::compute_minmax(){
//   has_minmax = 1;
//   if(data.empty()) {
//     min = 0;
//     max = 0;
//     return false;
//   }
//   else {
//     vector<T>::iterator itr;
//     T lmin = data[0];
//     T lmax = lmin;
//     for(itr = data.begin(); itr != data.end(); itr++){
//       lmin = Min(lmin, *itr);
//       lmax = Max(lmax, *itr);
//     }
//     min = (double) lmin;
//     max = (double) lmax;
//     return true;
//   }
// }

template <class T> void
DiscreteAttrib<T>::resize(int x, int y, int z)
{
  d_nx = x;
  d_ny = y;
  d_nz = z;
  d_dim = 3;
}


template <class T> void
DiscreteAttrib<T>::resize(int x, int y)
{
  d_nx = x;
  d_ny = y;
  d_nz = 0;
  d_dim = 2;
}


template <class T> void
DiscreteAttrib<T>::resize(int x)
{
  d_nx = x;
  d_ny = 0;
  d_nz = 0;
  d_dim = 1;
}


template <class T> int
DiscreteAttrib<T>::iterate(AttribFunctor<T> &func)
{
  func(d_defval);
  return 1;
}


template <class T> int
DiscreteAttrib<T>::size() const
{
  switch (d_dim)
    {
    case 1:
      return d_nx;

    case 2:
      return d_nx * d_ny;

    case 3:
      return d_nx * d_ny * d_nz;

    default:
      return 0;
    }
}

template <class T> string
DiscreteAttrib<T>::getInfo()
{
  ostringstream retval;
  retval <<
    "Name = " << d_name << '\n' <<
    "Type = DiscreteAttrib" << '\n' <<
    "Dim = " << d_dim << ": " << d_nx << ' ' << d_ny << ' ' << d_nz << '\n' <<
    "Size = " << size() << '\n';
  return retval.str();
}

template <class T> bool 
DiscreteAttrib<T>::isValid(int ix) const{
  ASSERTEQ(d_dim, 1);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  return testValidBit(ix);
}

template <class T> bool 
DiscreteAttrib<T>::isValid(int ix, int iy) const{
  ASSERTEQ(d_dim, 2);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  return testValidBit(iy*(d_nx)+ix);
}

template <class T> bool 
DiscreteAttrib<T>::isValid(int ix, int iy, int iz) const{
  ASSERTEQ(d_dim, 3);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  CHECKARRAYBOUNDS(iz, 0, d_nz);
  return testValidBit(iz*(d_nx*d_ny)+iy*(d_nx)+ix);
}


template <class T> int
DiscreteAttrib<T>::resizeValidBits(int nelem){
  int nallocNew = nelem/nbits+1;
  unsigned int* newbuff = 0;
  try {
    newbuff = new unsigned int[nallocNew];
  }
  catch (bad_alloc){
    dbg << "Memory allocation error in DiscreteAttrib<T>::resizeValidBits(int)" << endl;
    throw;
  }
  
  for (int i=0; i<nallocNew; i++)
    newbuff[i]=maxUI;
  
  if (d_pValidBits){
    int cpyNum = (nallocNew>d_nalloc)?d_nalloc:nallocNew;
    memcpy(newbuff, d_pValidBits, sizeof (unsigned int)*cpyNum);
    delete[] d_pValidBits;
  }

  d_pValidBits = newbuff;
  d_nalloc = nallocNew;

  return d_nalloc;
}

template <class T> inline bool
DiscreteAttrib<T>::testValidBit(int bitn) const{
  return ( d_pValidBits[bitn/nbits] >> (bitn%nbits)) & 1;
}

template <class T> inline void 
DiscreteAttrib<T>::validSet(int lb, int rb, bool bitVal){
  CHECKARRAYBOUNDS(lb, 0, d_nx*d_ny*d_nz);
  CHECKARRAYBOUNDS(rb, lb, d_nx*d_ny*d_nz);

  int lbInd = lb/nbits;
  int rbInd = rb/nbits;
  unsigned int lbMask = (bitVal)?(maxUI<<(lb%nbits)):~(maxUI<<(lb%nbits));
  unsigned int rbMask = (bitVal)?(~(maxUI<<((rb+1)%nbits))):(maxUI<<((rb+1)%nbits));

  if (lbInd==rbInd)
    if (bitVal)
      d_pValidBits[lbInd] |=(lbMask&rbMask);
    else
      d_pValidBits[lbInd] &=(lbMask&rbMask);
  else {
    if (bitVal){
      d_pValidBits[lbInd] |= lbMask;
      for (int i=lbInd+1; i<rbInd; i++){
	d_pValidBits[i] = maxUI;
      }
      d_pValidBits[rbInd] |= rbMask;
    }
    else {
      d_pValidBits[lbInd] &= lbMask; 
      for (int i=lbInd+1; i<rbInd; i++){
	d_pValidBits[i] = 0;
      }
      d_pValidBits[rbInd] &= rbMask;
    }
  }
}

template <class T> inline void 
DiscreteAttrib<T>::validSet(int pos, bool bitVal){
  CHECKARRAYBOUNDS(pos, 0, d_nx*d_ny*d_nz);

  unsigned int mask = 1<<(pos%nbits);
 
  if (bitVal)
    d_pValidBits[pos/nbits] |= mask;
  else
    d_pValidBits[pos/nbits] &= ~mask;
 
  for (int i=0; i<d_nalloc; i++)
    cout << hex << d_pValidBits[i] << " ";
  cout << endl;
}

template <class T> inline void
DiscreteAttrib<T>::setValidBit(int ix, bool bitVal){
  ASSERTEQ(d_dim, 1);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  validSet(ix, bitVal);
}

template <class T> inline void
DiscreteAttrib<T>::setValidBit(int ix, int iy, bool bitVal){
  ASSERTEQ(d_dim, 2);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  validSet(iy*(d_nx)+ix, bitVal);
}

template <class T> inline void
DiscreteAttrib<T>::setValidBit(int ix, int iy, int iz, bool bitVal){
  ASSERTEQ(d_dim, 3);
  CHECKARRAYBOUNDS(ix, 0, d_nx);
  CHECKARRAYBOUNDS(iy, 0, d_ny);
  CHECKARRAYBOUNDS(iz, 0, d_nz);
  validSet((iz*d_ny+iy)*d_nx+ix, bitVal);
}

template <class T> inline void
DiscreteAttrib<T>::setValidBit(const IntVector& low, const IntVector& high, bool bitVal){
  
  if (low.x()==high.x()){
    for (int i = low.x(); i<=high.x(); i++)
      for (int j = low.y(); j<=high.y(); j++)
	for (int k = low.z(); k<=high.z(); k++)
	  validSet((k*d_ny+j)*d_nx+i, bitVal);
  }
  else {
    int il = low.x(), ih = high.x();    
    for (int j=low.y(); j<=high.y(); j++)
      for (int k=low.z(); k<=high.z(); k++)
	validSet((k*d_ny+j)*d_nx+il, (k*d_ny+j)*d_nx+ih, bitVal);
  }
}

template <class T> int  
DiscreteAttrib<T>::getNumOfValid() const{
  int resNum = 0;

  if (d_pValidBits){
    unsigned int buff = 0;
    for (int i=0; i<d_nx*d_ny*d_nz; i++){
      
      if (!(i%nbits)){
	buff = d_pValidBits[i/nbits];
      }
     
      if(buff & 1){
	resNum++;
      }
      buff>>=1;
    }
  }
  
  return resNum;
}

template <class T> bool  
DiscreteAttrib<T>::getValid1(int ix, T& res){
  res = get1(ix);
  return testValidBit(ix);
}

template <class T> bool
DiscreteAttrib<T>::getValid2(int ix, int iy, T& res){
  res = get2(ix, iy);
  return testValidBit(iy*(d_nx)+ix);
}

template <class T> bool
DiscreteAttrib<T>::getValid3(int ix, int iy, int iz, T& res){
  res = get3(ix, iy, iz);
  return testValidBit(iz*(d_nx*d_ny)+iy*(d_nx)+ix);
}

template <class T> void
DiscreteAttrib<T>::initialize(const T& val){
  int i, j, k;
  if (d_dim==1)
    for (i=0; i<d_nx; i++)
      fset1(i, val);
  else if (d_dim==2)
    for (i=0; i<d_nx; i++)
      for (j=0; j<d_ny; j++)
	fset2(i, j, val);
  else
    for (k=0; k<d_nz; k++)
      for (j=0; j<d_ny; j++)
	for (i=0; i<d_nx; i++)
	  fset3(i, j, k, val); 
}

template <class T> const unsigned int*
DiscreteAttrib<T>::getValidBits(int& nalloc){
  nalloc = d_nalloc;
  return d_pValidBits;
}
 
template <class T> void 
DiscreteAttrib<T>::copyValidBits(const unsigned int* pValidBits, int nalloc){
  resizeValidBits(nalloc);
  memcpy(d_pValidBits, pValidBits, nalloc*sizeof(unsigned int));
}
  
} // End namespace SCIRun

#endif
