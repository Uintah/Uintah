/*
 *  ColumnMatrix.h: for RHS and LHS
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ColumnMatrix_h
#define SCI_project_ColumnMatrix_h 1

#include <Core/share/share.h>

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Datatypes/Matrix.h>

#include <iosfwd>  // Forward declarations for KCC C++ I/O routines

namespace SCIRun {

class SCICORESHARE ColumnMatrix : public Matrix {
  int rows;
  double* data;
public:
  double* get_rhs() const {return data;}
  void put_lhs(double* lhs) {data = lhs;} 

  ColumnMatrix(int rows=0);
  virtual ~ColumnMatrix();
  ColumnMatrix(const ColumnMatrix&);
  ColumnMatrix& operator=(const ColumnMatrix&);
  virtual ColumnMatrix* clone();
  inline double& operator[](int) const;

  virtual double& get(int, int);
  double& get(int);
  virtual void put(int, int, const double&);
  void put(int, const double&);
  virtual int nrows() const;
  virtual int ncols() const;
  virtual double minValue();
  virtual double maxValue();
  virtual void getRowNonzeros(int r, Array1<int>& idx, Array1<double>& val);
  virtual int solve(ColumnMatrix&);
  virtual int solve(vector<double>& sol);
  virtual void zero();
  virtual double sumOfCol(int);
  virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		    int& flops, int& memrefs, int beg=-1, int end=-1, 
		    int spVec=0) const;
  virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
			      int& flops, int& memrefs,
			      int beg=-1, int end=-1, int spVec=0);
  virtual void print();

  double vector_norm();
  double vector_norm(int& flops, int& memrefs);
  double vector_norm(int& flops, int& memrefs, int beg, int end);
  
  friend SCICORESHARE void Mult(ColumnMatrix&, const ColumnMatrix&, double s);
  friend SCICORESHARE void Mult(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);
  friend SCICORESHARE void Mult(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&,
				int& flops, int& memrefs);
  friend SCICORESHARE void Mult(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&,
				int& flops, int& memrefs, int beg, int end);
  friend SCICORESHARE void Sub(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);
  friend SCICORESHARE void Sub(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&,
			       int& flops, int& memrefs);
  friend SCICORESHARE double Dot(const ColumnMatrix&, const ColumnMatrix&);
  friend SCICORESHARE double Dot(const ColumnMatrix&, const ColumnMatrix&,
				 int& flops, int& memrefs);
  friend SCICORESHARE double Dot(const ColumnMatrix&, const ColumnMatrix&,
				 int& flops, int& memrefs, int beg, int end);
  friend SCICORESHARE void ScMult_Add(ColumnMatrix&, double s, const ColumnMatrix&,
				      const ColumnMatrix&);
  friend SCICORESHARE void ScMult_Add(ColumnMatrix&, double s, const ColumnMatrix&,
				      const ColumnMatrix&, int& flops, int& memrefs);
  friend SCICORESHARE void ScMult_Add(ColumnMatrix&, double s, const ColumnMatrix&,
				      const ColumnMatrix&, int& flops, int& memrefs,
				      int beg, int end);
  
  friend SCICORESHARE void Copy(ColumnMatrix&, const ColumnMatrix&);
  friend SCICORESHARE void Copy(ColumnMatrix&, const ColumnMatrix&, int& flops, int& refs,
				int beg, int end);
  friend SCICORESHARE void AddScMult(ColumnMatrix&, const ColumnMatrix&, double s, const ColumnMatrix&);
  friend SCICORESHARE void Add(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);
  friend SCICORESHARE void Add(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);
  
  void print(std::ostream&);
  void resize(int);
  
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};
} // End namespace SCIRun
#include <Core/Util/Assert.h>

namespace SCIRun {

inline double& ColumnMatrix::operator[](int i) const
{
    ASSERTRANGE(i, 0, rows)
    return data[i];
}

} // End namespace SCIRun

#endif
