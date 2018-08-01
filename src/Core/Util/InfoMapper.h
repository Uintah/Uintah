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

#include <sci_defs/visit_defs.h>

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
    return m_keys.size();
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
      m_values[i] = val;
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

  virtual bool exists( const unsigned int index ) const
  {
    return (index < m_keys.size());
  };

  virtual bool exists( const std::string name ) const
  {
    for (unsigned int i = 0; i < m_names.size(); ++i) {
      if( name == m_names[i] ) {
        return true;
      }
    }
    return false;
  };

  virtual void insert( const E key, const std::string name,
                                    const std::string units, const T value )
  {
    if (!exists(key) && !exists(name)) {
      unsigned int index = m_keys.size();
      m_keys[key] = index;
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

        T& operator[](const E key)       { return m_values[ m_keys[key] ]; };
  const T& operator[](const E key) const { return m_values[ m_keys[key] ]; };


        T& operator[](unsigned int index)       { return m_values[index]; };
  const T& operator[](unsigned int index) const { return m_values[index]; };

  // Set value
  virtual void setValue( const E key, const T value )
  {
    validKey( key );

    m_values[ m_keys[key] ] = value;
  };

  // Get value
  virtual T getRankValue( const E key )
  {
    validKey( key );

    return m_values[ m_keys[key] ];
  };

  virtual T getRankValue( const unsigned int index )
  {
    const E key = getKey(index);

    return getRankValue( key );
  };

  virtual T getRankValue( const std::string name )
  {
    const E key = getKey(name);

    return getRankValue( key );
  };

  // Get name
  virtual std::string getName( const E key )
  {
    validKey( key );

    return m_names[ m_keys[key] ];
  };

  virtual std::string getName( const unsigned int index )
  {
    const E key = getKey(index);

    return getName( key );
  };

  virtual std::string getName( const std::string name )
  {
    const E key = getKey(name);

    return getName( key );
  };

  // Get units
  virtual std::string getUnits( const E key )
  {
    validKey( key );

    return m_units[ m_keys[key] ];
  };

  virtual std::string getUnits( const unsigned int index )
  {
    const E key = getKey(index);

    return getUnits( key );
  };

  virtual std::string getUnits( const std::string name )
  {
    const E key = getKey(name);

    return getUnits( key );
  };

  // Get key
  virtual E getKey( const unsigned int index )
  {
    for (typename std::map< E, unsigned int >::iterator it=m_keys.begin();
         it!=m_keys.end(); ++it)
    {
      if( index == it->second )
        return it->first;
    }

    std::cerr << "InfoMapper::getKey bad index " << index << std::endl;

    return m_keys.begin()->first;
  };

  virtual E getKey( const std::string name )
  {
    for (unsigned int i = 0; i < m_names.size(); ++i) {
      if( name == m_names[ i ] ) {
        return getKey( i );
      }
    }

    std::cerr << "InfoMapper::getKey bad name " << name << std::endl;

    return m_keys.begin()->first;
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

  // Get the sum for all ranks on a single node
  virtual double getNodeSum( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_node_sum[ InfoMapper<E, T>::m_keys[key] ];
  };

  virtual double getNodeSum( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getNodeSum( key );
  };
  
  virtual double getNodeSum( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getNodeSum( key );
  };

  // Get the average for all ranks on a single node
  virtual double getNodeAverage( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_node_average[ InfoMapper<E, T>::m_keys[key] ];
  };

  virtual double getNodeAverage( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getNodeAverage( key );
  };
  
  virtual double getNodeAverage( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getNodeAverage( key );
  };

  // Get average over all ranks
  virtual double getRankAverage( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_rank_average[ InfoMapper<E, T>::m_keys[key] ];
  };

  virtual double getRankAverage( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getRankAverage( key );
  };

  virtual double getRankAverage( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getRankAverage( key );
  };

  // Get maxium over all ranks
  virtual double getRankMaximum( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_rank_maximum[ InfoMapper<E, T>::m_keys[key] ].val;
  };

  virtual double getRankMaximum( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getRankMaximum( key );
  };

  virtual double getRankMaximum( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getRankMaximum( key );
  };

  // Get rank
  virtual unsigned int getRankForMaximum( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_rank_maximum[ InfoMapper<E, T>::m_keys[key] ].rank;
  };

  virtual unsigned int getRankForMaximum( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getRankForMaximum( key );
  };

  virtual unsigned int getRankForMaximum( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getRankForMaximum( key );
  };

  // Reduce
  virtual void reduce( bool allReduce, const ProcessorGroup* myWorld )
  {
    unsigned int nStats = InfoMapper<E, T>::m_keys.size();

    if (nStats == 0) {
      return;
    }

    if (myWorld->nRanks() > 1) {
      m_node_sum.resize(nStats);
      m_node_average.resize(nStats);
      m_rank_average.resize(nStats);
      m_rank_maximum.resize(nStats);

      std::vector<double>      toReduce( nStats );
      std::vector<double_int>  toReduceMax( nStats );

      // Do the reductions across all ranks.

      // A little ugly, but do it anyway so only one reduction is needed
      // for the sum and one for the maximum. 
      for (size_t i = 0; i < nStats; ++i) {
        toReduce[i] = InfoMapper<E, T>::m_values[i];
        toReduceMax[i] =
	  double_int(InfoMapper<E, T>::m_values[i], myWorld->myRank());
      }

      // Reduction across each node.
#ifdef HAVE_VISIT
      Uintah::MPI::Allreduce(&toReduce[0], &m_node_sum[0], nStats,
			     MPI_DOUBLE, MPI_SUM, myWorld->getNodeComm());

      for (size_t i = 0; i < nStats; ++i) {
	m_node_average[i] = m_node_sum[i] / myWorld->myNode_nRanks();
      }      
#endif
      
      // Reductions across all ranks.
      if (allReduce) {
        Uintah::MPI::Allreduce(&toReduce[0], &m_rank_average[0], nStats,
			       MPI_DOUBLE, MPI_SUM, myWorld->getComm());
        Uintah::MPI::Allreduce(&toReduceMax[0], &m_rank_maximum[0], nStats,
			       MPI_DOUBLE_INT, MPI_MAXLOC, myWorld->getComm());
      }
      else {
        Uintah::MPI::Reduce(&toReduce[0], &m_rank_average[0], nStats,
			    MPI_DOUBLE, MPI_SUM, 0, myWorld->getComm());
        Uintah::MPI::Reduce(&toReduceMax[0], &m_rank_maximum[0], nStats,
			    MPI_DOUBLE_INT, MPI_MAXLOC, 0, myWorld->getComm());
      }

      // Calculate the averages.
      for (unsigned i = 0; i < nStats; ++i)
        m_rank_average[i] /= myWorld->nRanks();
    }

    // Single rank so just copy the values.
    else {
      m_node_sum.resize(nStats);
      m_node_average.resize(nStats);

      m_rank_average.resize(nStats);
      m_rank_maximum.resize(nStats);

      for (size_t i = 0; i < nStats; ++i) {
        double val = InfoMapper<E, T>::m_values[i];

        m_node_sum[i] = val;
        m_node_average[i] = val;
        m_rank_average[i] = val;
        m_rank_maximum[i] = double_int(val, 0);
      }
    }
  };

protected:

  std::vector< double > m_node_sum;     // Sum of all ranks on a single node
  std::vector< double > m_node_average; // Average of all ranks on a single node
  
  std::vector< double >     m_rank_average;      // Average over all ranks
  std::vector< double_int > m_rank_maximum;      // Maximum over all ranks
};

} // End namespace Uintah

#endif // CORE_UTIL_INFOMAPPER_H
