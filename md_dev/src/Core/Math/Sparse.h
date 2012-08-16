/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#ifndef __sparse_matrix_h
#define __sparse_matrix_h

#include <map>
#include <cassert>
#include <valarray>
#include <numeric>

namespace Uintah {

template<class ValueType, class IndexType, class ContainerType> 
class MatrixElement {
 private:
  ContainerType& C;
  typename ContainerType::iterator I;
  IndexType row, column;

 public:
  typedef std::pair<IndexType, IndexType> IndexPair;
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
  typedef std::pair<IndexType, IndexType> IndexPair;
  typedef std::map<IndexPair, ValueType, std::less<IndexPair> > ContainerType;
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

  std::valarray<ValueType> operator *(std::valarray<ValueType>& x) {
    assert((size_type)x.size() == Columns());
    std::valarray<ValueType> b(ValueType(0),Columns());
    for (typename SparseMatrix<ValueType,IndexType>::iterator itr = begin(); 
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

 std::valarray<double> cgSolve(SparseMatrix<double,int>& A, std::valarray<double>& b,
	  		          int conflag);

 double eigenvalue(SparseMatrix<double,int>& A, std::valarray<double>& eigenvector);

 double conditionNum(SparseMatrix<double,int>& A);

}

#endif
  











