#ifndef __sparse_matrix_h
#define __sparse_matrix_h

#include <map>
#include <assert.h>
#include <sgi_stl_warnings_off.h>
#include <valarray>
#include <numeric>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using namespace std;

template<class ValueType, class IndexType, class ContainerType> 
class MatrixElement {
 private:
  ContainerType& C;
  typename ContainerType::iterator I;
  IndexType row, column;

 public:
  typedef pair<IndexType, IndexType> IndexPair;
  typedef MatrixElement<ValueType, IndexType, ContainerType>& Reference;
  
  MatrixElement(ContainerType& Cont, IndexType r, IndexType c) : C(Cont),
    I(C.find(IndexPair(r,c))), row(r), column(c)
    {}

  ValueType asValue() const
    {
      if (I == C.end())
	return ValueType(0);
      else
	return (*I).second;
    }

  operator ValueType () const
    {
      return asValue();
    }

  Reference operator=(const ValueType& x)
  {
    if (x != ValueType(0) )
      {
	if (I == C.end()) {
	  assert(C.size() < C.max_size());
	  I = (C.insert(typename ContainerType::value_type(
			  IndexPair(row,column), x))
	       ).first;
	}
	else
	  (*I).second = x;
      }
    else
      if (I != C.end()) {
	C.erase(I);
	I = C.end();
      }
    return *this;
  }

  Reference operator=(Reference rhs)
  {
    if (this != &rhs) {
      return operator=(rhs.asValue());
    }
    return *this;
  }
};
  

template<class ValueType, class IndexType> class SparseMatrix {
 public:
  typedef pair<IndexType, IndexType> IndexPair;
  typedef map<IndexPair, ValueType, less<IndexPair> > ContainerType;
  typedef MatrixElement<ValueType, IndexType, ContainerType> ME;

  typedef IndexType size_type;

 private:
  size_type rows, columns;
  ContainerType C;

 public:
  SparseMatrix()  {}
  SparseMatrix(size_type r, size_type c) : rows(r), columns(c) {}
  size_type Rows() const { return rows;}
  size_type Columns() const { return columns; }
  
  typedef typename ContainerType::iterator iterator;
  typedef typename ContainerType::const_iterator const_iterator;

  size_type size() const { return C.size(); }
  size_type max_size() const {return C.max_size(); }

  void setSize(size_type r, size_type c) { rows=r; columns=c;}
  
  iterator begin() { return C.begin(); }
  iterator end() { return C.end(); }

  const_iterator begin() const { return C.begin();}
  const_iterator end() const { return C.end();}

  void clear() { C.clear(); };

  valarray<ValueType> operator *(valarray<ValueType>& x) {
    assert((size_type)x.size() == Columns());
    valarray<ValueType> b(ValueType(0),Columns());
    for (SparseMatrix<ValueType,IndexType>::iterator itr = begin(); 
	 itr != end(); itr++) {
      b[Index1(itr)] += Value(itr)*x[Index2(itr)];
    }
    return b;
  }


  class Aux {
  public:
    Aux(size_type r, size_type maxs, ContainerType& Cont) 
      : Row(r), maxColumns(maxs), C(Cont) {};
    
    ME operator[] (size_type c)
      {
	assert(c >= 0 && c < maxColumns);
	return ME(C, Row, c);
      }
    
  private:
    size_type Row, maxColumns;
    ContainerType& C;
  };
  
  Aux operator[](size_type r)
    {
      assert(r >= 0 && r < rows);
      return Aux(r, columns, C);
    };
  
  size_type Index1(iterator& I) const
    {
      return (*I).first.first;
    }
  
  size_type Index2(iterator& I) const
    {
      return (*I).first.second;
    }
  
  ValueType Value(iterator& I) const
    {
      return (*I).second;
    }
  
};

valarray<double> cgSolve(SparseMatrix<double,int>& A, valarray<double>& b,
			 int conflag);

double eigenvalue(SparseMatrix<double,int>& A, valarray<double>& eigenvector);

double conditionNum(SparseMatrix<double,int>& A);

}

#endif
  











