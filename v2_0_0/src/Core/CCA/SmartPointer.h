// TODO
// - get rid of reference/dereference, put in bodies?
// - STL operators?
// What to do with asserts
// documentation
// operator void*

/*
 *  CCALib/SmartPointer.h: Smart Pointers
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 */

// Requires methods:
//  void _addReference();
//  bool _deleteReference();
//  int getReferenceCount() const;

#ifndef CCALib_SmartPointer_h
#define CCALib_SmartPointer_h

//#include <cca_config.h>

namespace CCALib {
  template<typename T> class SmartPointer {
  public:
    typedef T element_type;

    explicit SmartPointer(T* ptr=0)
      : ptr(ptr) {
      reference();
    }

    ~SmartPointer() {
      dereference();
    }

    SmartPointer(const SmartPointer<T>& copy)
      : ptr(copy.ptr) {
      reference();
    }

    SmartPointer<T>& operator=(const SmartPointer<T>& copy) {
      copy.reference();
      dereference();
      ptr=copy.ptr;
      return *this;
    }

    SmartPointer<T>& operator=(T* inptr) {
      if(inptr)
	inptr->_addReference();
      dereference();
      ptr=inptr;
      return *this;
    }

    // Allow casting from a derived class
    template<typename D> SmartPointer(const SmartPointer<D>& copy)
      : ptr(copy.getPointer()) {
      reference();
    }

    // Allow = from a derived class
    template<typename D>
    SmartPointer<T>& operator=(const SmartPointer<D>& inptr) {
      inptr.reference();
      dereference();
      ptr=inptr.getPointer();
      return *this;
    }

    inline T* operator->() const {
      //ASSERT(ptr != 0);
      return ptr;
    }
    inline T& operator*() const {
      //ASSERT(ptr != 0);
      return ptr;
    }
    inline T* getPointer() const {
	return ptr;
    }

    int useCount() const {
      return ptr?ptr->getReferenceCount():0;
    }
    bool isUnique() const {
      return ptr?ptr->getReferenceCount() == 1:false;
      // Returns false if ptr is null.  Should it be true, or should it throw?
    }
    bool isNull() const {
      return ptr == 0;
    }
    void reference() const {
      if(ptr)
	ptr->_addReference();
    }
    void dereference() {
      if(ptr)
	ptr->_deleteReference();
    }
  private:
    // mutable allows reference() to be used on a const pointer
    mutable T* ptr;
  };

template<typename T>
  inline bool operator==(const SmartPointer<T>& a, const SmartPointer<T>& b)
    { return a.getPointer() == b.getPointer(); }

template<typename T>
  inline bool operator!=(const SmartPointer<T>& a, const SmartPointer<T>& b)
    { return a.getPointer() != b.getPointer(); }
} // End namespace CCA

#endif
