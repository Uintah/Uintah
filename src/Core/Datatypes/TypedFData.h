// TypedFData.h - the base field data class.
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   January 2001
//
//  Copyright (C) 2001 SCI Institute
//
//  General storage class for Fields.
//

#ifndef SCI_project_TypedFData_h
#define SCI_project_TypedFData_h 1

#include <Core/Datatypes/FData.h>


namespace SCIRun {

template <class T>
class TypedFData : public FData
{
public:
  typedef T value_type;

  // GROUP:  Constructors/Destructor
  //////////
  //
  TypedFData();
  virtual ~TypedFData();
  
  // GROUP: Class interface functions
  //////////
  // 

  virtual void get(T &result, int *loc) = 0;
  virtual void set(const T &val, int *loc) = 0;
  
  // GROUP: Support of persistent representation
  //////////
  //
  void    io(Piostream&);
  static  PersistentTypeID type_id;
  static  const string type_name(int);
  virtual const string get_type_name(int n) = 0;

protected:
};



template <class T>
class FData1D : public TypedFData<T>
{
public:
  // GROUP:  Constructors/Destructor
  //////////
  //
  FData1D();
  virtual ~FData1D();
  
  // GROUP: Class interface functions
  //////////
  // 
  
  // GROUP: Support of persistent representation
  //////////
  //
  void    io(Piostream&);
  static  PersistentTypeID type_id;
  static  const string type_name(int);
  virtual const string get_type_name(int n) = 0;

  virtual void get(T &result, int *loc) { fget(result, loc); }
  virtual void set(const T &val, int *loc) { fset(val, loc); }

  void fget(T &result, int *loc) { result = container[*loc]; }
  void fset(const T &val, int *loc) { container[*loc] = val; }

protected:
  Array1<T> container;
};



template <class T, class R, class F>
class FDataUnOp : public TypedFData<T>
{
public:
  FDataUnOp();
  virtual ~FDataUnOp();

  void    io(Piostream&);
  static  PersistentTypeID type_id;
  static  const string type_name(int);
  virtual const string get_type_name(int n) = 0;

  virtual void get(T &result, int *loc) { fget(result, loc); }
  virtual void set(const T &val, int *loc) { fset(val, loc); }

  void fget(T &result, int *loc)
  {
    R::value_type tmp;
    fdata->fget(tmp, loc);
    F f; f(result, tmp);
  }

  void fset(const T &, int *)
  {
    // Not implemented.  Functional FData is read only.
  }

protected:
  
  Handle<R> fdata;
};


template <class T, class R1, class R2, class F>
class FDataBinOp : public TypedFData<T>
{
public:
  FDataBinOp();
  virtual ~FDataBinOp();

  void    io(Piostream&);
  static  PersistentTypeID type_id;
  static  const string type_name(int);
  virtual const string get_type_name(int n) = 0;

  virtual void get(T &result, int *loc) { fget(result, loc); }
  virtual void set(const T &val, int *loc) { fset(val, loc); }

  void fget(T &result, int *loc)
  {
    R1::value_type tmp1;
    fdata1->fget(tmp1, loc);

    R2::value_type tmp2;
    fdata2->fget(tmp2, loc);

    F f; f(result, tmp1, tmp2);
  }

  void fset(const T &, int *)
  {
    // Not implemented.  Functional FData is read only.
  }

protected:
  Handle<R1> fdata1;
  Handle<R2> fdata2;
};  
  
}  // end namespace SCIRun

#endif
