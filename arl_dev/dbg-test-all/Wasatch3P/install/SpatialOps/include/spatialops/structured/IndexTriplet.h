/*
 * Copyright (c) 2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef SpatialOps_IndexTriplet_h
#define SpatialOps_IndexTriplet_h

/**
 *  \file   IndexTriplet.h
 *
 *  \date   Sep 29, 2011
 *  \author James C. Sutherland
 */

#include <spatialops/SpatialOpsConfigure.h>
#include <spatialops/SpatialOpsTools.h>
#include <spatialops/SpatialOpsDefs.h>

#include <spatialops/structured/IntVec.h>

#include <iomanip>
#include <string>
#include <sstream>
#include <stdexcept>

namespace SpatialOps{

  /**
   *  \struct Abs
   *  \brief Obtain the absolute value of an integer at compile time
   *
   *  Examples:
   *  \code assert( Abs<-4>::result == 4 ); \endcode
   */
  template< int I > struct Abs{ enum{ result = (I>=0) ? I : -I }; };

  /**
   * \struct Max
   * \brief  Select the maximum of two integer values
   *
   * Examples:
   * \code assert( Max<-1,3>::result == 3 ); \endcode
   */
  template< int i1, int i2 > struct Max{ enum{ result = i1>i2 ? i1 : i2 }; };

  /**
   * \struct Min
   * \brief  Select the minimum of two integer values
   *
   * Examples:
   * \code assert( Min<1,3>::result == 1 );
   * \endcode
   */
  template< int i1, int i2 > struct Min{ enum{ result = i1<i2 ? i1 : i2 }; };

  /**
   *  \struct LThan
   *  \brief Obtain the result of comparing two integers at compile time
   *
   *  Examples:
   *  \code assert( LThan<0, 1>::result == 1 ); \endcode
   *  \code assert( LThan<1, 0>::result == 0 ); \endcode
   *  \code assert( LThan<1, 1>::result == 0 ); \endcode
   */
  template< int i1, int i2 > struct LThan{ enum{ result = (i1 < i2) ? 1 : 0 }; };

  /**
   *  \struct GThan
   *  \brief Obtain the result of comparing two integers at compile time
   *
   *  Examples:
   *  \code assert( GThan<0, 1>::result == 0 ); \endcode
   *  \code assert( GThan<1, 0>::result == 1 ); \endcode
   *  \code assert( GThan<1, 1>::result == 0 ); \endcode
   */
  template< int i1, int i2 > struct GThan{ enum{ result = (i1 > i2) ? 1 : 0 }; };

  /**
   *  \struct IndexTriplet
   *  \brief Used for specifying field type traits.
   *
   *  \par Key Definitions
   *  \li \c X The x-component of the IndexTriplet
   *  \li \c Y The y-component of the IndexTriplet
   *  \li \c Z The z-component of the IndexTriplet
   *  \li \c Abs The absolute value of this IndexTriplet
   *
   *  Examples:
   *  \code
   *  typedef IndexTriplet<1,2,3> T;
   *  assert( T::X == 1 );
   *  assert( T::Y == 2 );
   *  assert( T::Z == 3 );
   *  \endcode
   *  \code
   *  typedef typename IndexTriplet<-1,2,0>::Abs  T2
   *  assert( T2::X==1 );
   *  assert( T2::Y==2 );
   *  assert( T2::Z==0 );
   *  \endcode
   */
  template< int i1, int i2, int i3 >
  struct IndexTriplet
  {
    enum Component{
      X=i1,  ///< The "x" component of the IndexTriplet
      Y=i2,  ///< The "y" component of the IndexTriplet
      Z=i3   ///< The "z" component of the IndexTriplet
    };

    /**
     * \typedef Abs
     * \brief The absolute value of this IndexTriplet
     */
    typedef IndexTriplet< Abs<i1>::result, Abs<i2>::result, Abs<i3>::result >  Abs;

    /**
     * \typedef PositiveOrZero
     * \brief   Preserves positive elements, zeros others.
     */
    typedef IndexTriplet< Max<0,i1>::result, Max<0,i2>::result, Max<0,i3>::result >  PositiveOrZero;

    /**
     * \typedef Negate
     * \brief Negates this IndexTriplet
     *  Example:
     *  \code
     *    typedef IndexTriplet< 1,-1, 1>               T1;
     *    typedef IndexTriplet<-1, 1,-1>               T2;
     *    typedef Negate<T1>::result                   T3;  // same as T2
     *    typedef Negate<T3>::result                   T4;  // same as T1
     *    typedef Negate< Negate<T1>::result >::result T5;  // same as T1 and T3
     *   \endcode
     */
    typedef IndexTriplet< -i1, -i2, -i3 > Negate;

    /**
     * \brief Writes the IndexTriplet to a string.
     * \return a string value representing the IndexTriplet.
     */
    static std::string print(){
      std::stringstream s;
      s << "("
          << std::setw(2) << X << ","
          << std::setw(2) << Y << ","
          << std::setw(2) << Z << " )";
      return s.str();
    }

    static inline int x_value(){ return int(X); }
    static inline int y_value(){ return int(Y); }
    static inline int z_value(){ return int(Z); }

    static inline int value(int const direction) {
      if(direction == 0)
        return int(X);
      else if(direction == 1)
        return int(Y);
      else if(direction == 2)
        return int(Z);
      else {
        std::ostringstream msg;
        msg << "IndexTriplet value() given bad direction; given: " << direction << "\n";
        throw(std::runtime_error(msg.str()));
      }
    };

    static inline IntVec int_vec(){
      return IntVec( i1, i2, i3 );
    }

#   ifdef __CUDACC__
    __device__ static inline int x_value_gpu() { return int(X); }
    __device__ static inline int y_value_gpu() { return int(Y); }
    __device__ static inline int z_value_gpu() { return int(Z); }

    __device__ static inline int value_gpu(int const direction)
    {
      if     ( direction == 0 ) return int(X);
      else if( direction == 1 ) return int(Y);
      else if( direction == 2 ) return int(Z);
      else return 9001; //Error: Cannot throw an error on a GPU
    }
#   endif
  };


  /**
   *  \struct UnitTriplet
   *  \brief Obtain the "unit vector" IndexTriplet for the supplied direction
   *  \tparam DirT (XDIR,YDIR,ZDIR)
   *  Examples:
   *  \code
   *    typedef UnitTriplet<XDIR>::type  XTriplet;  // same as IndexTriplet<1,0,0>;
   *    typedef UnitTriplet<YDIR>::type  YTriplet;  // same as IndexTriplet<0,1,0>;
   *    typedef UnitTriplet<ZDIR>::type  ZTriplet;  // same as IndexTriplet<0,0,1>;
   *  \endcode
   */
  template< typename DirT > struct UnitTriplet{
    typedef IndexTriplet< IsSameType<DirT,SpatialOps::XDIR>::result,
        IsSameType<DirT,SpatialOps::YDIR>::result,
        IsSameType<DirT,SpatialOps::ZDIR>::result
        > type;
  };


  /**
   *  \struct IndexTripletExtract
   *  \brief Extracts the value for the IndexTriplet in the requested dimension
   *
   *  Example usage:
   *   \code assert( IndexTripletExtract< IndexTriplet<-1,0,1>, XDIR >::value == -1 ); \endcode
   *   \code typedef IndexTripletExtract< IndexTriplet<-1,0,1>, XDIR >  ExtractedIndexType; \endcode
   *
   *  Note that only specialized versions of this struct exist for
   *  querying in valid dimensions (XDIR,YDIR,ZDIR).
   */
  template< typename IT, typename DirT > struct IndexTripletExtract;

  template< int i1, int i2, int i3 > struct IndexTripletExtract<IndexTriplet<i1,i2,i3>,XDIR>{ enum{ value = i1 }; };
  template< int i1, int i2, int i3 > struct IndexTripletExtract<IndexTriplet<i1,i2,i3>,YDIR>{ enum{ value = i2 }; };
  template< int i1, int i2, int i3 > struct IndexTripletExtract<IndexTriplet<i1,i2,i3>,ZDIR>{ enum{ value = i3 }; };
  template< int i1, int i2, int i3 > struct IndexTripletExtract<IndexTriplet<i1,i2,i3>,NODIR>;  // invalid


  /**
   *  \struct Kronecker
   *  \brief Implements a Kronecker delta function on two int values.
   *
   *  Examples:
   *  \code assert( Kronecker<1,0>::value == 0 ); \endcode
   *  \code assert( Kronecker<2,2>::value == 1 ); \endcode
   */
  template< int i1, int i2 > struct Kronecker       { enum{value=0}; };
  template< int i1         > struct Kronecker<i1,i1>{ enum{value=1}; };


  /**
   *  \struct IndexStagger
   *
   *  \brief Obtain the index value for how far the given field type
   *         is staggered relative to a scalar cell centered variable.
   *         Nominally 0 or -1.
   *
   *  \tparam FieldT the type of field in consideration.
   *  \tparam DirT   the direction we are interested in.
   *
   *  Example usage:
   *  \code
   *    const int n = IndexStagger<FieldT,DirT>::value;
   *  \endcode
   */
  template< typename FieldT, typename DirT >
  struct IndexStagger{
    enum{ value = IndexTripletExtract<typename FieldT::Location::Offset,DirT>::value };
  };

  //------------------------------------------------------------------

  /**
   *  \struct Add
   *  \brief Perform compile-time addition of two IndexTriplet types
   *
   *  Example usage:
   *   \code
   *    // In the following, MyResult and T1 are the same type:
   *    typedef IndexTriplet< 0, 0, 0> T1;
   *    typedef IndexTriplet< 1,-1, 1> T2;
   *    typedef IndexTriplet<-1, 1,-1> T3;
   *    typedef Add< T2, T3 >::result  MyResult;
   *   \endcode
   */
  template< typename IX1, typename IX2 >
  struct Add
  {
    typedef IndexTriplet< IX1::X + IX2::X,
        IX1::Y + IX2::Y,
        IX1::Z + IX2::Z >  result;
  };

  /**
   *  \struct Subtract
   *  \brief Perform compile-time subtraction of two IndexTriplet types
   *
   *  Example usage:
   *   \code
   *    // In the following, MyResult and T3 are the same type:
   *    typedef IndexTriplet< 0, 0, 0> T1;
   *    typedef IndexTriplet< 1,-1, 1> T2;
   *    typedef IndexTriplet<-1, 1,-1> T3;
   *    typedef Subtract< T1, T2 >::result  MyResult;
   *   \endcode
   */
  template< typename IX1, typename IX2 >
  struct Subtract
  {
    typedef IndexTriplet< IX1::X - IX2::X,
        IX1::Y - IX2::Y,
        IX1::Z - IX2::Z >  result;
  };

  /**
   *  \struct Multiply
   *  \brief Perform compile-time multiplication of two IndexTriplet types
   *
   *  Example usage:
   *   \code
   *    // In the following, MyResult and T3 are the same type:
   *    typedef IndexTriplet<-1,-1,-1> T1;
   *    typedef IndexTriplet< 1,-1, 1> T2;
   *    typedef IndexTriplet<-1, 1,-1> T3;
   *    typedef Multiply< T1, T2 >::result  MyResult;
   *   \endcode
   */
  template< typename IX1, typename IX2 >
  struct Multiply
  {
    typedef IndexTriplet< IX1::X * IX2::X,
        IX1::Y * IX2::Y,
        IX1::Z * IX2::Z >  result;
  };

  /**
   *  \struct LessThan
   *  \brief Perform compile-time compare of two IndexTriplet types
   *
   *  Example usage:
   *   \code
   *    // In the following, MyResult and T3 are the same type:
   *    typedef IndexTriplet<-1, 1,-1> T1;
   *    typedef IndexTriplet< 1,-1, 1> T2;
   *    typedef IndexTriplet< 1, 0, 1> T3;
   *    typedef LessThan< T1, T2 >::result  MyResult;
   *   \endcode
   */
  template< typename IX1, typename IX2 >
  struct LessThan
  {
    typedef IndexTriplet< LThan<IX1::X, IX2::X>::result,
        LThan<IX1::Y, IX2::Y>::result,
        LThan<IX1::Z, IX2::Z>::result > result;
  };

  /**
   *  \struct GreaterThan
   *  \brief Perform compile-time compare of two IndexTriplet types
   *
   *  Example usage:
   *   \code
   *    // In the following, MyResult and T3 are the same type:
   *    typedef IndexTriplet<-1, 1,-1> T1;
   *    typedef IndexTriplet< 1,-1, 1> T2;
   *    typedef IndexTriplet< 0, 1, 0> T3;
   *    typedef GreaterThan< T1, T2 >::result  MyResult;
   *   \endcode
   */
  template< typename IX1, typename IX2 >
  struct GreaterThan
  {
    typedef IndexTriplet< GThan<IX1::X, IX2::X>::result,
        GThan<IX1::Y, IX2::Y>::result,
        GThan<IX1::Z, IX2::Z>::result > result;
  };

  /**
   *  \struct GetNonzeroDir
   *
   *  \brief Assuming that only a single direction is active (1) in
   *         the IndexTriplet type, this identifies the direction via
   *         the typedef \c Dir.
   */
  template< typename IT > struct GetNonzeroDir;
  template<> struct GetNonzeroDir< IndexTriplet< 1, 0, 0> >{ typedef SpatialOps::XDIR  DirT; };
  template<> struct GetNonzeroDir< IndexTriplet<-1, 0, 0> >{ typedef SpatialOps::XDIR  DirT; };
  template<> struct GetNonzeroDir< IndexTriplet< 0, 1, 0> >{ typedef SpatialOps::YDIR  DirT; };
  template<> struct GetNonzeroDir< IndexTriplet< 0,-1, 0> >{ typedef SpatialOps::YDIR  DirT; };
  template<> struct GetNonzeroDir< IndexTriplet< 0, 0, 1> >{ typedef SpatialOps::ZDIR  DirT; };
  template<> struct GetNonzeroDir< IndexTriplet< 0, 0,-1> >{ typedef SpatialOps::ZDIR  DirT; };
  template<> struct GetNonzeroDir< IndexTriplet< 0, 0, 0> >{ typedef SpatialOps::NODIR DirT; };

} // namespace SpatialOps

/**
 * @}
 */

#endif /* SpatialOps_IndexTriplet_h */
