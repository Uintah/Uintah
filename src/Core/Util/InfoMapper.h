/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef CORE_UTIL_INFOMAPPER_H
#define CORE_UTIL_INFOMAPPER_H

#include <Core/Exceptions/InternalError.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahMPI.h>

#include <iostream>
#include <sstream>
#include <map>
#include <vector>

namespace Uintah {

struct double_int
{
  double val;
  int rank;
  double_int(double val, int rank): val(val), rank(rank) {}
  double_int(): val(0), rank(-1) {}
};


template<class E, class T>class InfoMapper
{
public:
  InfoMapper()
  {
    d_values.clear();
    d_names.clear();
    d_units.clear();
  };

  virtual ~InfoMapper() {};

  virtual size_t size() const
  {
    return d_values.size();
  };

  virtual E lastKey() const
  {
    return (E) d_values.size();
  };

  virtual void clear()
  {
    d_values.clear();
    d_names.clear();
    d_units.clear();
  };

  virtual void reset( const T val )
  {
    for( unsigned int i=0; i<d_values.size(); ++i ) {
      d_values[(E) i] = val;
    }
  };

  virtual void validKey( const E key ) const
  {
    if( !exists( key ) )
    {
      std::stringstream msg;
      msg << "Requesting an undefined key (" << key << ") ";
      throw Uintah::InternalError( msg.str(), __FUNCTION__, __LINE__);
    }
  };

  virtual bool exists( const E key ) const
  {
    return (d_keys.find( key ) != d_keys.end());
  };

  virtual bool exists( const std::string name )
  {
    for( unsigned int i=0; i<d_names.size(); ++i )
    {
      if( name == d_names[(E) i] ) {
        return true;
      }
    }
    return false;
  };

  virtual void validate( const E lastKey ) const
  {
    if( d_values.size() != (unsigned int) lastKey )
    {
      std::stringstream msg;
      msg << "The count does not match. Expected "
	        << (unsigned int) lastKey << " values. But added "
	        << d_values.size() << " values.";
      throw Uintah::InternalError(msg.str(), __FILE__, __LINE__);
    }
  };

  virtual void insert( const E key, const std::string name,
		                   const std::string units, const T value )
  {
    if( !exists( key ) && !exists( name ) && (unsigned int) key == d_keys.size() )
    {
      d_keys[key] = (unsigned int) key;
      d_values.push_back( value );
      d_names.push_back( name );
      d_units.push_back( units );
    }
    else
    {
      std::stringstream msg;
      msg << "Adding a key (" << key << ") with name, "
	        << name << " that already exists.";
      throw Uintah::InternalError( msg.str(), __FUNCTION__, __LINE__);
    }
  };

  // void erase( const E key )
  // {
  //   typename std::map< E, T >::iterator           vIter = d_values.find( key );
  //   typename std::map< E, std::string >::iterator nIter = d_names.find( key );
  //   typename std::map< E, std::string >::iterator uIter = d_units.find( key );

  //   if( vIter != d_values.end() &&
  // 	nIter != d_names.end() && uIter != d_units.end() )
  //   {
  //     d_values.erase(key);
  //     d_names.erase(key);
  //     d_units.erase(key);
  //   }
  //   else
  //   {
  //     std::stringstream msg;
  //     msg << "Trying to delete a key (" << key << ") that does not exist.";
  //     throw Uintah::InternalError( msg.str(), __FUNCTION__, __LINE__);
  //   }
  // }

        T& operator[](E idx)       { return d_values[idx]; };
  const T& operator[](E idx) const { return d_values[idx]; };

  virtual void setValue( const E key, const T value )
  {
    d_values[key] = value;
  };

  virtual T getValue( const E key )
  {
    validKey( key );

    return d_values[key];
  };

  virtual T getValue( const std::string name )
  {
    E key = getKey(name);

    validKey( key );

    return d_values[ key ];
  };

  virtual std::string getName( const E key )
  {
    validKey( key );

    return d_names[ key ];
  };

  virtual std::string getUnits( const E key )
  {
    validKey( key );

    return d_units[ key ];
  };

  virtual E getKey( const std::string name )
  {
    for( unsigned int i=0; i<d_names.size(); ++i )
    {
      if( name == d_names[(E) i] ) {
        return (E) i;
      }
    }

    return (E) d_values.size();
  };

protected:
  std::map< E, unsigned int > d_keys;
  std::vector< T >            d_values;
  std::vector< std::string >  d_names;
  std::vector< std::string >  d_units;
};


template<class E, class T> class ReductionInfoMapper : public InfoMapper<E, T>
{
public:
  ReductionInfoMapper()
  {
    d_average.clear();
    d_maximum.clear();
  };

  virtual ~ReductionInfoMapper() {};

  virtual void clear()
  {
    InfoMapper<E, T>::clear();

    d_average.clear();
    d_maximum.clear();
  };

  virtual void insert( const E key, const std::string name,
		       const std::string units, const T value )
  {
    InfoMapper<E, T>::insert( key, name, units, value );

    d_average.push_back(-1);
    d_maximum.push_back(double_int(0,-1));
  }

  virtual double getAverage( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return d_average[ key ];
  };

  virtual double getMaximum( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return d_maximum[ key ].val;
  };

  virtual unsigned int getRank( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return d_maximum[ key ].rank;
  };

  virtual void reduce( bool allReduce, const ProcessorGroup* myWorld )
  {
    unsigned int nStats = InfoMapper<E, T>::d_keys.size();

    if( nStats == 0 )
      return;
    
    if( myWorld->size() > 1)
    {
      // A little ugly, but do it anyway so only one reduction is needed
      // for the sum and one for the maximum. 
      std::vector<double>      toReduce( nStats );
      std::vector<double_int>  toReduceMax( nStats );

      d_average.resize( nStats );
      d_maximum.resize( nStats );

      for (size_t i=0; i<nStats; ++i)
      {
        toReduce[i] = InfoMapper<E, T>::d_values[i];
        toReduceMax[i] = double_int( InfoMapper<E, T>::d_values[i], myWorld->myrank() );
      }

      if( allReduce )
      {
        Uintah::MPI::Allreduce( &toReduce[0],    &d_average[0], nStats, MPI_DOUBLE,     MPI_SUM,    myWorld->getComm() );
        Uintah::MPI::Allreduce( &toReduceMax[0], &d_maximum[0], nStats, MPI_DOUBLE_INT, MPI_MAXLOC, myWorld->getComm() );
      }
      else
      {
        Uintah::MPI::Reduce( &toReduce[0],    &d_average[0], nStats, MPI_DOUBLE,     MPI_SUM,    0, myWorld->getComm() );
        Uintah::MPI::Reduce( &toReduceMax[0], &d_maximum[0], nStats, MPI_DOUBLE_INT, MPI_MAXLOC, 0, myWorld->getComm() );
      }

      // make sums averages
      for (unsigned i = 0; i < nStats; ++i)
      {
        d_average[i] /= myWorld->size();
      }
    }
    else
    {
      d_average.resize( nStats );
      d_maximum.resize( nStats );

      for (size_t i=0; i<nStats; ++i)
      {
        d_average[i] = InfoMapper<E, T>::d_values[i];
        d_maximum[i] = double_int(InfoMapper<E, T>::d_values[i], 0);
      }
    }
  };

protected:
  std::vector< double >     d_average;
  std::vector< double_int > d_maximum;
};

} // End namespace Uintah

#endif // CORE_UTIL_INFOMAPPER_H
