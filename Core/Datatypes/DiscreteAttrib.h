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
  typedef T value_type;

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
  virtual string getTypeName(int=0);
  
  //////////
  // Persistent representation...
  virtual void io(Piostream &);
  static PersistentTypeID type_id;
  static string typeName(int);
  static Persistent* maker();

  virtual int iterate(AttribFunctor<T> &func);

  virtual int xsize() const { return nx_; }
  virtual int ysize() const { return ny_; }
  virtual int zsize() const { return nz_; }
  virtual int dimension() const { return dim_; }
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
  int nx_, ny_, nz_;
  int dim_;

  unsigned int*   pValidBits_;
  int             nalloc_;
 
private:
  T defval_;
  static DebugStream dbg;
};

//////////
// Static members for PIO support
template <class T> 
DebugStream DiscreteAttrib<T>::dbg("DiscreteAttrib", true);
 

template <class T>
string DiscreteAttrib<T>::typeName(int n=0){
  ASSERTRANGE(n, 0, 2);
  static string t1name    = findTypeName((T*)0);
  static string className = string("DiscreteAttrib<") + t1name +">";
  
  switch (n){
  case 1:
    return t1name;
  default:
    return className;
  }
}

template <class T> Persistent*
DiscreteAttrib<T>::maker(){
  return new DiscreteAttrib<T>(0);
}

#ifdef __sgi

template <class T> 
const unsigned int DiscreteAttrib<T>::maxUI = numeric_limits<unsigned int>::max();

#else

template <class T> 
const unsigned int DiscreteAttrib<T>::maxUI = 0xFFFF;

#endif


template <class T> 
PersistentTypeID DiscreteAttrib<T>::type_id(DiscreteAttrib<T>::typeName(0), 
					    Attrib::typeName(0), 
					    DiscreteAttrib<T>::maker);

#define DISCRETEATTRIB_VERSION 1

template <class T> void
DiscreteAttrib<T>::io(Piostream& stream)
{
  stream.begin_class(typeName(0).c_str(), DISCRETEATTRIB_VERSION);
  
  // -- base class PIO
  Attrib::io(stream);
  Pio(stream, nx_);
  Pio(stream, ny_);
  Pio(stream, nz_);
  Pio(stream, dim_);
  Pio(stream, defval_);
  Pio(stream, nalloc_);
  
  if(stream.reading()){
    delete[] pValidBits_;        // -- in case we're reading different file into the object
    pValidBits_ = new unsigned int[nalloc_];
  }
  
  for (int i=0; i<nalloc_; i++)
    Pio(stream, pValidBits_[i]);
  
  stream.end_class();
}

//////////
// Constructors/Destructor
template <class T>
DiscreteAttrib<T>::DiscreteAttrib(int ix) :
  Attrib(), nx_(ix), ny_(0), nz_(0), dim_(1)
{
  pValidBits_ = 0;
  resizeValidBits(ix);
}

template <class T>
DiscreteAttrib<T>::DiscreteAttrib(int ix, int iy) :
  Attrib(), nx_(ix), ny_(iy), nz_(0), dim_(2)
{
  pValidBits_ = 0;
  resizeValidBits(ix*iy);
}

template <class T>
DiscreteAttrib<T>::DiscreteAttrib(int ix, int iy, int iz) :
  Attrib(), nx_(ix), ny_(iy), nz_(iz), dim_(3)
{
  pValidBits_ = 0;
  resizeValidBits(ix*iy*iz);
}

template <class T>
DiscreteAttrib<T>::DiscreteAttrib(const DiscreteAttrib& copy) :
  Attrib(copy),
  nx_(copy.nx_), ny_(copy.ny_), nz_(copy.nz_),
  dim_(copy.dim_)
{
  pValidBits_ = 0;
  resizeValidBits(nx_*ny_*nz_);
  memcpy(pValidBits_, copy.pValidBits_, nalloc_*sizeof(unsigned int));
}

template <class T>
DiscreteAttrib<T>::~DiscreteAttrib()
{
  delete[] pValidBits_;
}


template <class T> T &
DiscreteAttrib<T>::fget1(int)
{
  return defval_;
}


template <class T> T &
DiscreteAttrib<T>::fget2(int, int)
{
  return defval_;
}


template <class T> T &
DiscreteAttrib<T>::fget3(int, int, int)
{
  return defval_;
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
  defval_ = val;
}


template <class T> void
DiscreteAttrib<T>::fset2(int, int, const T &val)
{
  defval_ = val;
}


template <class T> void
DiscreteAttrib<T>::fset3(int, int, int, const T &val)
{
  defval_ = val;
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

template <class T> int
DiscreteAttrib<T>::iterate(AttribFunctor<T> &func)
{
  func(defval_);
  return 1;
}


template <class T> int
DiscreteAttrib<T>::size() const
{
  switch (dim_)
    {
    case 1:
      return nx_;

    case 2:
      return nx_ * ny_;

    case 3:
      return nx_ * ny_ * nz_;

    default:
      return 0;
    }
}

template <class T> string
DiscreteAttrib<T>::getInfo()
{
  ostringstream retval;
  retval <<
    "Name = " << name_ << '\n' <<
    "Type = DiscreteAttrib" << '\n' <<
    "Dim = " << dim_ << ": " << nx_ << ' ' << ny_ << ' ' << nz_ << '\n' <<
    "Size = " << size() << '\n';
  return retval.str();
}

template <class T> bool 
DiscreteAttrib<T>::isValid(int ix) const{
  ASSERTEQ(dim_, 1);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  return testValidBit(ix);
}

template <class T> bool 
DiscreteAttrib<T>::isValid(int ix, int iy) const{
  ASSERTEQ(dim_, 2);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  CHECKARRAYBOUNDS(iy, 0, ny_);
  return testValidBit(iy*(nx_)+ix);
}

template <class T> bool 
DiscreteAttrib<T>::isValid(int ix, int iy, int iz) const{
  ASSERTEQ(dim_, 3);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  CHECKARRAYBOUNDS(iy, 0, ny_);
  CHECKARRAYBOUNDS(iz, 0, nz_);
  return testValidBit(iz*(nx_*ny_)+iy*(nx_)+ix);
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
  
  if (pValidBits_){
    int cpyNum = (nallocNew>nalloc_)?nalloc_:nallocNew;
    memcpy(newbuff, pValidBits_, sizeof (unsigned int)*cpyNum);
    delete[] pValidBits_;
  }

  pValidBits_ = newbuff;
  nalloc_ = nallocNew;

  return nalloc_;
}

template <class T> inline bool
DiscreteAttrib<T>::testValidBit(int bitn) const{
  return ( pValidBits_[bitn/nbits] >> (bitn%nbits)) & 1;
}

template <class T> inline void 
DiscreteAttrib<T>::validSet(int lb, int rb, bool bitVal){
  CHECKARRAYBOUNDS(lb, 0, nx_*ny_*nz_);
  CHECKARRAYBOUNDS(rb, lb, nx_*ny_*nz_);

  int lbInd = lb/nbits;
  int rbInd = rb/nbits;
  unsigned int lbMask = (bitVal)?(maxUI<<(lb%nbits)):~(maxUI<<(lb%nbits));
  unsigned int rbMask = (bitVal)?(~(maxUI<<((rb+1)%nbits))):(maxUI<<((rb+1)%nbits));

  if (lbInd==rbInd)
    if (bitVal)
      pValidBits_[lbInd] |=(lbMask&rbMask);
    else
      pValidBits_[lbInd] &=(lbMask&rbMask);
  else {
    if (bitVal){
      pValidBits_[lbInd] |= lbMask;
      for (int i=lbInd+1; i<rbInd; i++){
	pValidBits_[i] = maxUI;
      }
      pValidBits_[rbInd] |= rbMask;
    }
    else {
      pValidBits_[lbInd] &= lbMask; 
      for (int i=lbInd+1; i<rbInd; i++){
	pValidBits_[i] = 0;
      }
      pValidBits_[rbInd] &= rbMask;
    }
  }
}

template <class T> inline void 
DiscreteAttrib<T>::validSet(int pos, bool bitVal){
  CHECKARRAYBOUNDS(pos, 0, nx_*ny_*nz_);

  unsigned int mask = 1<<(pos%nbits);
 
  if (bitVal)
    pValidBits_[pos/nbits] |= mask;
  else
    pValidBits_[pos/nbits] &= ~mask;
 
  for (int i=0; i<nalloc_; i++)
    cout << hex << pValidBits_[i] << " ";
  cout << endl;
}

template <class T> inline void
DiscreteAttrib<T>::setValidBit(int ix, bool bitVal){
  ASSERTEQ(dim_, 1);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  validSet(ix, bitVal);
}

template <class T> inline void
DiscreteAttrib<T>::setValidBit(int ix, int iy, bool bitVal){
  ASSERTEQ(dim_, 2);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  CHECKARRAYBOUNDS(iy, 0, ny_);
  validSet(iy*(nx_)+ix, bitVal);
}

template <class T> inline void
DiscreteAttrib<T>::setValidBit(int ix, int iy, int iz, bool bitVal){
  ASSERTEQ(dim_, 3);
  CHECKARRAYBOUNDS(ix, 0, nx_);
  CHECKARRAYBOUNDS(iy, 0, ny_);
  CHECKARRAYBOUNDS(iz, 0, nz_);
  validSet((iz*ny_+iy)*nx_+ix, bitVal);
}

template <class T> inline void
DiscreteAttrib<T>::setValidBit(const IntVector& low, const IntVector& high, bool bitVal){
  
  if (low.x()==high.x()){
    for (int i = low.x(); i<=high.x(); i++)
      for (int j = low.y(); j<=high.y(); j++)
	for (int k = low.z(); k<=high.z(); k++)
	  validSet((k*ny_+j)*nx_+i, bitVal);
  }
  else {
    int il = low.x(), ih = high.x();    
    for (int j=low.y(); j<=high.y(); j++)
      for (int k=low.z(); k<=high.z(); k++)
	validSet((k*ny_+j)*nx_+il, (k*ny_+j)*nx_+ih, bitVal);
  }
}

template <class T> int  
DiscreteAttrib<T>::getNumOfValid() const{
  int resNum = 0;

  if (pValidBits_){
    unsigned int buff = 0;
    for (int i=0; i<nx_*ny_*nz_; i++){
      
      if (!(i%nbits)){
	buff = pValidBits_[i/nbits];
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
  return testValidBit(iy*(nx_)+ix);
}

template <class T> bool
DiscreteAttrib<T>::getValid3(int ix, int iy, int iz, T& res){
  res = get3(ix, iy, iz);
  return testValidBit(iz*(nx_*ny_)+iy*(nx_)+ix);
}

template <class T> void
DiscreteAttrib<T>::initialize(const T& val){
  int i, j, k;
  if (dim_==1)
    for (i=0; i<nx_; i++)
      fset1(i, val);
  else if (dim_==2)
    for (i=0; i<nx_; i++)
      for (j=0; j<ny_; j++)
	fset2(i, j, val);
  else
    for (k=0; k<nz_; k++)
      for (j=0; j<ny_; j++)
	for (i=0; i<nx_; i++)
	  fset3(i, j, k, val); 
}

template <class T> const unsigned int*
DiscreteAttrib<T>::getValidBits(int& nalloc){
  nalloc = nalloc_;
  return pValidBits_;
}
 
template <class T> void 
DiscreteAttrib<T>::copyValidBits(const unsigned int* pValidBits, int nalloc){
  resizeValidBits(nalloc);
  memcpy(pValidBits_, pValidBits, nalloc*sizeof(unsigned int));
}

template <class T> string
DiscreteAttrib<T>::getTypeName(int n){
  return typeName(n);
}
  
} // End namespace SCIRun

#endif
