/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
#include <map>
#include <sstream>
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
    m_values.clear();
    m_names.clear();
    m_units.clear();
  };

  virtual ~InfoMapper() {};

  virtual size_t size() const
  {
    return m_values.size();
  };

  virtual E lastKey() const
  {
    return (E) m_values.size();
  };

  virtual void clear()
  {
    m_values.clear();
    m_names.clear();
    m_units.clear();
  };

  virtual void reset( const T val )
  {
    for (unsigned int i = 0; i < m_values.size(); ++i) {
      m_values[(E)i] = val;
    }
  };

  virtual void validKey( const E key ) const
  {
    if (!exists(key)) {
      std::stringstream msg;
      msg << "Requesting an undefined key (" << key << ") ";
      throw Uintah::InternalError( msg.str(), __FUNCTION__, __LINE__);
    }
  };

  virtual bool exists( const E key ) const
  {
    return (m_keys.find( key ) != m_keys.end());
  };

  virtual bool exists( const std::string name ) const
  {
    for (unsigned int i = 0; i < m_names.size(); ++i) {
      if( name == m_names[(E) i] ) {
        return true;
      }
    }
    return false;
  };

  virtual void validate( const E lastKey ) const
  {
    if (m_values.size() != (unsigned int)lastKey) {
      std::stringstream msg;
      msg << "The count does not match. Expected "
	        << (unsigned int) lastKey << " values. But added "
	        << m_values.size() << " values.";
      throw Uintah::InternalError(msg.str(), __FILE__, __LINE__);
    }
  };

  virtual void insert( const E key, const std::string name,
		                   const std::string units, const T value )
  {
    if (!exists(key) && !exists(name) && (unsigned int)key == m_keys.size()) {
      m_keys[key] = (unsigned int)key;
      m_values.push_back(value);
      m_names.push_back(name);
      m_units.push_back(units);
    }
    else {
      std::stringstream msg;
      msg << "Adding a key (" << key << ") with name, " << name << " that already exists.";
      throw Uintah::InternalError( msg.str(), __FUNCTION__, __LINE__);
    }
  };

  // void erase( const E key )
  // {
  //   typename std::map< E, T >::iterator           vIter = m_values.find( key );
  //   typename std::map< E, std::string >::iterator nIter = m_names.find( key );
  //   typename std::map< E, std::string >::iterator uIter = m_units.find( key );

  //   if( vIter != m_values.end() &&
  // 	nIter != m_names.end() && uIter != m_units.end() )
  //   {
  //     m_values.erase(key);
  //     m_names.erase(key);
  //     m_units.erase(key);
  //   }
  //   else
  //   {
  //     std::stringstream msg;
  //     msg << "Trying to delete a key (" << key << ") that does not exist.";
  //     throw Uintah::InternalError( msg.str(), __FUNCTION__, __LINE__);
  //   }
  // }

        T& operator[](E idx)       { return m_values[idx]; };
  const T& operator[](E idx) const { return m_values[idx]; };

  virtual void setValue( const E key, const T value )
  {
    m_values[key] = value;
  };

  virtual T getValue( const E key ) const
  {
    validKey( key );

    return m_values[key];
  };

  virtual T getValue( const std::string name ) const
  {
    E key = getKey(name);

    validKey( key );

    return m_values[ key ];
  };

  virtual std::string getName( const E key ) const
  {
    validKey( key );

    return m_names[ key ];
  };

  virtual std::string getUnits( const E key ) const
  {
    validKey( key );

    return m_units[ key ];
  };

  virtual E getKey( const std::string name ) const
  {
    for (unsigned int i = 0; i < m_names.size(); ++i) {
      if( name == m_names[(E) i] ) {
        return (E) i;
      }
    }

    return (E) m_values.size();
  };

protected:
  std::map< E, unsigned int > m_keys;
  std::vector< T >            m_values;
  std::vector< std::string >  m_names;
  std::vector< std::string >  m_units;
};


template<class E, class T> class ReductionInfoMapper : public InfoMapper<E, T>
{
public:
  ReductionInfoMapper()
  {
    m_rank_average.clear();
    m_rank_maximum.clear();
  };

  virtual ~ReductionInfoMapper() {};

  virtual void clear()
  {
    InfoMapper<E, T>::clear();

    m_rank_average.clear();
    m_rank_maximum.clear();
  };

  virtual void insert( const E key, const std::string name,
		       const std::string units, const T value )
  {
    InfoMapper<E, T>::insert( key, name, units, value );

    m_rank_average.push_back(-1);
    m_rank_maximum.push_back(double_int(0,-1));
  }

  // getSum
  virtual double getSum( const E key ) const
  {
    InfoMapper<E, T>::validKey( key );

    return m_node_sum[ key ];
  };

  virtual double getSum( const std::string name ) const
  {
    E key = InfoMapper<E, T>::getKey(name);

    InfoMapper<E, T>::validKey( key );

    return m_node_sum[ key ];
  };

  // getAverage
  virtual double getAverage( const E key ) const
  {
    InfoMapper<E, T>::validKey( key );

    return m_rank_average[ key ];
  };

  virtual double getAverage( const std::string name ) const
  {
    E key = InfoMapper<E, T>::getKey(name);

    InfoMapper<E, T>::validKey( key );

    return m_rank_average[ key ];
  };

  // getMaxium
  virtual double getMaximum( const E key ) const
  {
    InfoMapper<E, T>::validKey( key );

    return m_rank_maximum[ key ].val;
  };

  virtual double getMaximum( const std::string name ) const
  {
    E key = InfoMapper<E, T>::getKey(name);

    InfoMapper<E, T>::validKey( key );

    return m_rank_maximum[ key ].val;
  };

  // getRank
  virtual unsigned int getRank( const E key ) const
  {
    InfoMapper<E, T>::validKey( key );

    return m_rank_maximum[ key ].rank;
  };

  virtual unsigned int getRank( const std::string name ) const
  {
    E key = InfoMapper<E, T>::getKey(name);

    InfoMapper<E, T>::validKey( key );

    return m_rank_maximum[ key ].rank;
  };

  // reduce
  virtual void reduce( bool allReduce, const ProcessorGroup* myWorld )
  {
    unsigned int nStats = InfoMapper<E, T>::m_keys.size();

    if (nStats == 0) {
      return;
    }

    if (myWorld->nRanks() > 1) {
      m_node_sum.resize(nStats);
      m_rank_average.resize(nStats);
      m_rank_maximum.resize(nStats);

      std::vector<double>      reduced( nStats );
      std::vector<double>      toReduce( nStats );
      std::vector<double_int>  toReduceMax( nStats );

      // Perform the reduction acrosss each processor node.
      for (int n = 0; n < myWorld->nNodes(); ++n) {
        // If this rank belongs to this node then pass the value.
        if (n == myWorld->myNode()) {
          for (size_t i = 0; i < nStats; ++i) {
            toReduce[i] = InfoMapper<E, T>::m_values[i];
          }
        }
        // This rank is not on the current node so ignore the values.
        else {
          for (size_t i = 0; i < nStats; ++i) {
            toReduce[i] = 0;
          }
        }
	
        Uintah::MPI::Allreduce( &toReduce[0], &reduced[0], nStats, MPI_DOUBLE, MPI_SUM, myWorld->getComm() );

	      // If this rank belongs to this node then save the summation values.
        if (n == myWorld->myNode()) {
          for (size_t i = 0; i < nStats; ++i) {
            m_node_sum[i] = reduced[i];
          }
        }
      }

      // Do the reductions across all ranks.

      // A little ugly, but do it anyway so only one reduction is needed
      // for the sum and one for the maximum. 
      for (size_t i = 0; i < nStats; ++i) {
        toReduce[i] = InfoMapper<E, T>::m_values[i];
        toReduceMax[i] = double_int(InfoMapper<E, T>::m_values[i], myWorld->myRank());
      }

      if (allReduce) {
        Uintah::MPI::Allreduce(&toReduce[0], &m_rank_average[0], nStats, MPI_DOUBLE, MPI_SUM, myWorld->getComm());
        Uintah::MPI::Allreduce(&toReduceMax[0], &m_rank_maximum[0], nStats, MPI_DOUBLE_INT, MPI_MAXLOC, myWorld->getComm());
      }
      else {
        Uintah::MPI::Reduce(&toReduce[0], &m_rank_average[0], nStats, MPI_DOUBLE, MPI_SUM, 0, myWorld->getComm());
        Uintah::MPI::Reduce(&toReduceMax[0], &m_rank_maximum[0], nStats, MPI_DOUBLE_INT, MPI_MAXLOC, 0, myWorld->getComm());
      }

      // Calculate the averages.
      for (unsigned i = 0; i < nStats; ++i)
        m_rank_average[i] /= myWorld->nRanks();
    }

    // Single rank so just copy the values.
    else {
      m_node_sum.resize(nStats);

      m_rank_average.resize(nStats);
      m_rank_maximum.resize(nStats);

      for (size_t i = 0; i < nStats; ++i) {
        double val = InfoMapper<E, T>::m_values[i];

        m_node_sum[i] = val;
        m_rank_average[i] = val;
        m_rank_maximum[i] = double_int(val, 0);
      }
    }
  };

protected:

  std::vector< double >     m_node_sum;
  std::vector< double >     m_rank_average;
  std::vector< double_int > m_rank_maximum;
};

} // End namespace Uintah

#endif // CORE_UTIL_INFOMAPPER_H
