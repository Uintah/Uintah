/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <Core/Util/DOUT.hpp>

#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
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

class BaseInfoMapper
{
public:

  enum OutputTypeEnum {
    Dout,
    Write_Last,
    Write_Append,
    Write_Separate
  };
};

////////////////////////////////////////////////////////////////////////////////
// The basic info mapper
template<class E, class T>class InfoMapper
{
  template<class e, class t>
    friend class VectorInfoMapper;
  template<class key, class e, class t>
    friend class MapInfoMapper;

public:

  InfoMapper()
  {
    m_values.clear();
    m_counts.clear();
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
    m_counts.clear();
    m_names.clear();
    m_units.clear();
  };

  virtual void reset( const T val )
  {
    for (unsigned int i = 0; i < m_values.size(); ++i) {
      m_values[i] = val;
      m_counts[i] = 0;
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
                                    const std::string units )
  {
    if (!exists(key) && !exists(name)) {
      unsigned int index = m_keys.size();
      m_keys[key] = index;
      m_values.push_back(0);
      m_counts.push_back(0);
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
    m_counts[ m_keys[key] ] = 0;
  };

  // Get value
  virtual T getRankValue( const E key )
  {
    validKey( key );

    if( m_counts[ m_keys[key] ] )
      return m_values[ m_keys[key] ] / m_counts[ m_keys[key] ];
    else
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

  // Get count
  virtual unsigned int getCount( const E key )
  {
    validKey( key );

    return m_counts[ m_keys[key] ];
  };

  virtual unsigned int getCount( const unsigned int index )
  {
    const E key = getKey(index);

    return getCount( key );
  };

  virtual unsigned int getCount( const std::string name )
  {
    const E key = getKey(name);

    return getCount( key );
  };

  // Increment count
  virtual unsigned int incrCount( const E key )
  {
    validKey( key );

    return ++m_counts[ m_keys[key] ];
  };

  virtual unsigned int incrCount( const unsigned int index )
  {
    const E key = getKey(index);

    return incrCount( key );
  };

  virtual unsigned int incrCount( const std::string name )
  {
    const E key = getKey(name);

    return incrCount( key );
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
  std::vector< unsigned int > m_counts;
  std::vector< std::string >  m_names;
  std::vector< std::string >  m_units;
};


////////////////////////////////////////////////////////////////////////////////
// The base reduction info mapper across all ranks on a node and all
// ranks utilized.
  template<class E, class T> class ReductionInfoMapper : public BaseInfoMapper, public InfoMapper<E, T>
{
public:

  ReductionInfoMapper()
  {
    m_rank_average.clear();
    m_rank_minimum.clear();
    m_rank_maximum.clear();
    m_rank_std_dev.clear();
  };

  virtual ~ReductionInfoMapper() {};

  // All ranks
  virtual void calculateRankSum    ( bool val ) { m_rank_calculate_sum     = val; }
  virtual void calculateRankAverage( bool val ) { m_rank_calculate_average = val; }
  virtual void calculateRankMinimum( bool val ) { m_rank_calculate_minimum = val; }
  virtual void calculateRankMaximum( bool val ) { m_rank_calculate_maximum = val; }
  virtual void calculateRankStdDev ( bool val ) { m_rank_calculate_std_dev = val; }

  virtual bool calculateRankSum    () const { return m_rank_calculate_sum;     }
  virtual bool calculateRankAverage() const { return m_rank_calculate_average; }
  virtual bool calculateRankMinimum() const { return m_rank_calculate_minimum; }
  virtual bool calculateRankMaximum() const { return m_rank_calculate_maximum; }
  virtual bool calculateRankStdDev () const { return m_rank_calculate_std_dev; }

  // All ranks on a node
  virtual void calculateNodeSum    ( bool val ) { m_node_calculate_sum     = val; }
  virtual void calculateNodeAverage( bool val ) { m_node_calculate_average = val; }
  virtual void calculateNodeMinimum( bool val ) { m_node_calculate_minimum = val; }
  virtual void calculateNodeMaximum( bool val ) { m_node_calculate_maximum = val; }
  virtual void calculateNodeStdDev ( bool val ) { m_node_calculate_std_dev = val; }

  virtual bool calculateNodeSum    () const { return m_node_calculate_sum; }
  virtual bool calculateNodeAverage() const { return m_node_calculate_average; }
  virtual bool calculateNodeMinimum() const { return m_node_calculate_minimum; }
  virtual bool calculateNodeMaximum() const { return m_node_calculate_maximum; }
  virtual bool calculateNodeStdDev () const { return m_node_calculate_std_dev; }

  virtual void clear()
  {
    InfoMapper<E, T>::clear();

    m_rank_average.clear();
    m_rank_minimum.clear();
    m_rank_maximum.clear();
    m_rank_std_dev.clear();

    m_node_sum.clear();
    m_node_average.clear();
    m_node_minimum.clear();
    m_node_maximum.clear();
    m_node_std_dev.clear();
  };

  virtual void insert( const E key, const std::string name,
                       const std::string units )
  {
    InfoMapper<E, T>::insert( key, name, units );

    m_rank_average.push_back(0);
    m_rank_minimum.push_back(double_int(0,-1));
    m_rank_maximum.push_back(double_int(0,-1));
    m_rank_std_dev.push_back(0);

    m_node_sum.push_back(0);
    m_node_average.push_back(0);
    m_node_minimum.push_back(double_int(0,-1));
    m_node_maximum.push_back(double_int(0,-1));
    m_node_std_dev.push_back(0);
  }

  // All ranks.

  // Get average over all ranks
  virtual double getRankSum( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_rank_sum[ InfoMapper<E, T>::m_keys[key] ];
  };

  virtual double getRankSum( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getRankSum( key );
  };

  virtual double getRankSum( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getRankSum( key );
  };

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

  // Get minium over all ranks
  virtual double getRankMinimum( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_rank_minimum[ InfoMapper<E, T>::m_keys[key] ].val;
  };

  virtual double getRankMinimum( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getRankMinimum( key );
  };

  virtual double getRankMinimum( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getRankMinimum( key );
  };

  // Get minimum rank
  virtual unsigned int getRankForMinimum( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_rank_minimum[ InfoMapper<E, T>::m_keys[key] ].rank;
  };

  virtual unsigned int getRankForMinimum( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getRankForMinimum( key );
  };

  virtual unsigned int getRankForMinimum( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getRankForMinimum( key );
  };

  // Get maximum over all ranks
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

  // Get maximmum rank
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

  // Get std dev over all ranks
  virtual double getRankStdDev( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_rank_std_dev[ InfoMapper<E, T>::m_keys[key] ];
  };

  virtual double getRankStdDev( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getRankStdDev( key );
  };

  virtual double getRankStdDev( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getRankStdDev( key );
  };

  // All ranks on a node
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

  // Get minium over all ranks on a single node
  virtual double getNodeMinimum( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_node_minimum[ InfoMapper<E, T>::m_keys[key] ].val;
  };

  virtual double getNodeMinimum( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getNodeMinimum( key );
  };

  virtual double getNodeMinimum( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getNodeMinimum( key );
  };

  // Get the rank for the minimum for a single node
  virtual unsigned int getNodeForMinimum( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_node_minimum[ InfoMapper<E, T>::m_keys[key] ].rank;
  };

  virtual unsigned int getNodeForMinimum( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getNodeForMinimum( key );
  };

  virtual unsigned int getNodeForMinimum( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getNodeForMinimum( key );
  };

  // Get maximum over all ranks on a single node
  virtual double getNodeMaximum( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_node_maximum[ InfoMapper<E, T>::m_keys[key] ].val;
  };

  virtual double getNodeMaximum( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getNodeMaximum( key );
  };

  virtual double getNodeMaximum( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getNodeMaximum( key );
  };

  // Get the rank for the maximum for a single node
  virtual unsigned int getNodeForMaximum( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_node_maximum[ InfoMapper<E, T>::m_keys[key] ].rank;
  };

  virtual unsigned int getNodeForMaximum( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getNodeForMaximum( key );
  };

  virtual unsigned int getNodeForMaximum( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getNodeForMaximum( key );
  };

  // Get std dev over all ranks on a single node
  virtual double getNodeStdDev( const E key )
  {
    InfoMapper<E, T>::validKey( key );

    return m_node_std_dev[ InfoMapper<E, T>::m_keys[key] ];
  };

  virtual double getNodeStdDev( const unsigned int index )
  {
    const E key = InfoMapper<E, T>::getKey(index);

    return getNodeStdDev( key );
  };

  virtual double getNodeStdDev( const std::string name )
  {
    const E key = InfoMapper<E, T>::getKey(name);

    return getNodeStdDev( key );
  };

  // Reduce
  virtual void reduce( bool allReduce, const ProcessorGroup* myWorld )
  {
    unsigned int nStats = InfoMapper<E, T>::m_keys.size();

    if (nStats == 0) {
      return;
    }

    if (myWorld->nRanks() > 1) {
      m_rank_sum.resize(nStats);
      m_rank_average.resize(nStats);
      m_rank_minimum.resize(nStats);
      m_rank_maximum.resize(nStats);
      m_rank_std_dev.resize(nStats);

      m_node_sum.resize(nStats);
      m_node_average.resize(nStats);
      m_node_minimum.resize(nStats);
      m_node_maximum.resize(nStats);
      m_node_std_dev.resize(nStats);

      std::vector<double>      toReduce( nStats );
      std::vector<double_int>  toReduceMin( nStats );
      std::vector<double_int>  toReduceMax( nStats );
      std::vector<double>      toReduceStdDev( nStats );

      // Do the reductions across all ranks.

      // A little ugly, but do it anyway so only one reduction is needed
      // for the sum, std. dev., minimum, and maximum (four total)
      for (size_t i = 0; i < nStats; ++i) {
        double val;
        if( InfoMapper<E, T>::m_counts[i] )
          val = InfoMapper<E, T>::m_values[i] / InfoMapper<E, T>::m_counts[i];
        else
          val = InfoMapper<E, T>::m_values[i];

        toReduce[i] = val;
        toReduceMin[i] = double_int( val, myWorld->myRank());
        toReduceMax[i] = double_int( val, myWorld->myRank());
      }

      // All ranks.

      // Sum reductions across all ranks.
      if( m_rank_calculate_sum || m_rank_calculate_average || m_rank_calculate_std_dev )
      {
        if (allReduce || m_rank_calculate_std_dev ) {
          Uintah::MPI::Allreduce(&toReduce[0], &m_rank_sum[0], nStats,
                                 MPI_DOUBLE, MPI_SUM, myWorld->getComm());
        }
        else {
          Uintah::MPI::Reduce(&toReduce[0], &m_rank_sum[0], nStats,
                              MPI_DOUBLE, MPI_SUM, 0, myWorld->getComm());
        }

        // Calculate the averages.
        for (size_t i = 0; i < nStats; ++i) {
          m_rank_average[i] = m_rank_sum[i] / myWorld->nRanks();
        }

        if( m_rank_calculate_std_dev )
        {
          //  Calculate the squared differences
          for (size_t i = 0; i < nStats; ++i) {
            double val = toReduce[i] - m_rank_average[i];
            toReduceStdDev[i] = val * val;
          }

          // Sum of squared differences reductions across all ranks.
          if (allReduce) {
            Uintah::MPI::Allreduce(&toReduceStdDev[0], &m_rank_std_dev[0], nStats,
                                   MPI_DOUBLE, MPI_SUM, myWorld->getComm());
          }
          else {
            Uintah::MPI::Reduce(&toReduceStdDev[0], &m_rank_std_dev[0], nStats,
                                MPI_DOUBLE, MPI_SUM, 0, myWorld->getComm());
          }

          // Calculate the std. dev.
          for (size_t i = 0; i < nStats; ++i) {
            if(myWorld->nRanks()-1 > 0)
              m_rank_std_dev[i] = std::sqrt(m_rank_std_dev[i] / (myWorld->nRanks()-1) );
            else
              m_rank_std_dev[i] = 0;
          }
        }
      }

      // Min reductions across all ranks.
      if( m_rank_calculate_minimum )
      {
        if (allReduce) {
          Uintah::MPI::Allreduce(&toReduceMin[0], &m_rank_minimum[0], nStats,
                                 MPI_DOUBLE_INT, MPI_MINLOC, myWorld->getComm());
        }
        else {
          Uintah::MPI::Reduce(&toReduceMin[0], &m_rank_minimum[0], nStats,
                              MPI_DOUBLE_INT, MPI_MINLOC, 0, myWorld->getComm());
        }
      }

      // Max reductions across all ranks.
      if( m_rank_calculate_maximum )
      {
        if (allReduce) {
          Uintah::MPI::Allreduce(&toReduceMax[0], &m_rank_maximum[0], nStats,
                                 MPI_DOUBLE_INT, MPI_MAXLOC, myWorld->getComm());
        }
        else {
          Uintah::MPI::Reduce(&toReduceMax[0], &m_rank_maximum[0], nStats,
                              MPI_DOUBLE_INT, MPI_MAXLOC, 0, myWorld->getComm());
        }
      }

      // All ranks on a node.

      // Sum reductions across each nodes.
      if( m_node_calculate_sum ||
          m_node_calculate_average || m_node_calculate_std_dev )
      {
        if (allReduce || m_node_calculate_std_dev ) {
          Uintah::MPI::Allreduce(&toReduce[0], &m_node_sum[0], nStats,
                                 MPI_DOUBLE, MPI_SUM, myWorld->getNodeComm());
        }
        else {
          Uintah::MPI::Reduce(&toReduce[0], &m_node_sum[0], nStats,
                              MPI_DOUBLE, MPI_SUM, 0, myWorld->getNodeComm());
        }

        // Calculate the averages.
        for (size_t i = 0; i < nStats; ++i) {
          m_node_average[i] = m_node_sum[i] / myWorld->myNode_nRanks();
        }

        if( m_node_calculate_std_dev )
        {
          //  Calculate the squared differences
          for (size_t i = 0; i < nStats; ++i) {
            double val = toReduce[i] - m_node_average[i];
            toReduceStdDev[i] = val * val;
          }

          // Sum of squared differences reductions across all nodes.
          if (allReduce) {
            Uintah::MPI::Allreduce(&toReduceStdDev[0], &m_node_std_dev[0], nStats,
                                   MPI_DOUBLE, MPI_SUM, myWorld->getNodeComm());
          }
          else {
            Uintah::MPI::Reduce(&toReduceStdDev[0], &m_node_std_dev[0], nStats,
                                MPI_DOUBLE, MPI_SUM, 0, myWorld->getNodeComm());
          }

          // Calculate the std. dev.
          for (size_t i = 0; i < nStats; ++i) {
            if(myWorld->myNode_nRanks()-1)
              m_node_std_dev[i] = std::sqrt(m_node_std_dev[i] / (myWorld->myNode_nRanks()-1) );
            else
              m_node_std_dev[i] = 0;
          }
        }
      }

      // Min reductions across each nodes.
      if( m_node_calculate_minimum )
      {
        if (allReduce) {
          Uintah::MPI::Allreduce(&toReduceMin[0], &m_node_minimum[0], nStats,
                                 MPI_DOUBLE_INT, MPI_MINLOC, myWorld->getNodeComm ());
        }
        else {
          Uintah::MPI::Reduce(&toReduceMin[0], &m_node_minimum[0], nStats,
                              MPI_DOUBLE_INT, MPI_MINLOC, 0, myWorld->getNodeComm());
        }
      }

      // Max reductions across each nodes.
      if( m_node_calculate_maximum )
      {
        if (allReduce) {
          Uintah::MPI::Allreduce(&toReduceMax[0], &m_node_maximum[0], nStats,
                                 MPI_DOUBLE_INT, MPI_MAXLOC, myWorld->getNodeComm());
        }
        else {
          Uintah::MPI::Reduce(&toReduceMax[0], &m_node_maximum[0], nStats,
                              MPI_DOUBLE_INT, MPI_MAXLOC, 0, myWorld->getNodeComm());
        }
      }
    }

    // Single rank so just copy the values.
    else {
      m_rank_average.resize(nStats);
      m_rank_minimum.resize(nStats);
      m_rank_maximum.resize(nStats);
      m_rank_std_dev.resize(nStats);

      m_node_sum.resize(nStats);
      m_node_average.resize(nStats);
      m_node_minimum.resize(nStats);
      m_node_maximum.resize(nStats);
      m_node_std_dev.resize(nStats);

      for (size_t i = 0; i < nStats; ++i) {
        double val;

        if(InfoMapper<E, T>::m_counts[i] )
          val = InfoMapper<E, T>::m_values[i] / InfoMapper<E, T>::m_counts[i];
        else
          val = InfoMapper<E, T>::m_values[i];

        m_rank_average[i] = val;
        m_rank_minimum[i] = double_int(val, 0);
        m_rank_maximum[i] = double_int(val, 0);
        m_rank_std_dev[i] = 0;

        m_node_sum[i] = val;
        m_node_average[i] = val;
        m_node_minimum[i] = double_int(val, 0);
        m_node_maximum[i] = double_int(val, 0);
        m_node_std_dev[i] = 0;
      }
    }
  };

  //______________________________________________________________________
  void reportRankSummaryStats( const std::string statsName,
                               const std::string preamble,
                               const int rank,
                               const int nRanks,
                               const int timeStep,
                               const double simTime,
                               const OutputTypeEnum oType,
                               bool calcImbalance )
  {
    unsigned int nStats = InfoMapper<E, T>::m_keys.size();

    if( nStats == 0 )
      return;

    // Get the max string length so to have descent formatting.
    std::ostringstream tmp;
    tmp << nRanks;

    unsigned int maxRankStrLength = std::max( (int) 4, (int) tmp.str().size() );
    unsigned int maxStatStrLength = std::string("Description").size();
    unsigned int maxUnitStrLength = 0;

    for (unsigned int i=0; i<nStats; ++i) {
      if (InfoMapper<E, T>::getRankValue(i) > 0) {

        if ( maxStatStrLength < InfoMapper<E, T>::getName(i).size() )
          maxStatStrLength = InfoMapper<E, T>::getName(i).size();

        if ( maxUnitStrLength < InfoMapper<E, T>::getUnits(i).size() )
          maxUnitStrLength = InfoMapper<E, T>::getUnits(i).size();
      }
    }

    // All ranks
    if( m_rank_calculate_sum ||
        m_rank_calculate_average || m_rank_calculate_std_dev ||
        m_rank_calculate_minimum || m_rank_calculate_maximum ) {

      bool m_calcImbalance = calcImbalance &&
        m_rank_calculate_average && m_rank_calculate_maximum;

      std::ostringstream header;

      if( !preamble.empty() )
        header << preamble << std::endl;

      header << statsName << " summary stats for "
             << "time step " << timeStep
             << " at time="  << simTime
             << std::endl
             << "  " << std::left
             << std::setw(maxStatStrLength+2) << "Description"
             << std::setw(maxUnitStrLength+5) << "Units";
      if (m_rank_calculate_sum) {
        std::ostringstream tmp;
        tmp << "Sum (" << nRanks << ")";
        header << std::setw(18) << tmp.str();
      }
      if (m_rank_calculate_minimum) {
        header << std::setw(18) << "Minimum"
               << std::setw(maxRankStrLength+3) << "Rank";
      }
      if (m_rank_calculate_average) {
        std::ostringstream tmp;
        tmp << "Average (" << nRanks << ")";
        header << std::setw(18) << tmp.str();
      }
      if (m_rank_calculate_std_dev) {
        header << std::setw(18) << "Std. Dev.";
      }
      if (m_rank_calculate_maximum) {
        header << std::setw(18) << "Maximum"
               << std::setw(maxRankStrLength+3) << "Rank";
      }
      if( m_calcImbalance ) {
        header << std::setw(12) << "100*(1-ave/max) '% load imbalance'";
      }

      header << std::endl;

      std::ostringstream message;

      for (unsigned int i=0; i<InfoMapper<E, T>::size(); ++i) {
        if( getRankAverage(i) != 0.0 || getRankMinimum(i) != 0.0 || getRankMaximum(i) != 0.0 ) {
          if (message.str().size()) {
            message << std::endl;
          }

          message << "  " << std::left << std::setw(maxStatStrLength+2)
                  << InfoMapper<E, T>::getName(i) << "[" << std::setw(maxUnitStrLength)
                  << InfoMapper<E, T>::getUnits(i) << "]";

          if (m_rank_calculate_sum) {
            message << " : " << std::setw(15) << getRankSum(i);
          }
          if (m_rank_calculate_minimum) {
            message << " : " << std::setw(15) << getRankMinimum(i)
                    << " : " << std::setw(maxRankStrLength)  << getRankForMinimum(i);
          }
          if (m_rank_calculate_average) {
            message << " : " << std::setw(15) << getRankAverage(i);
          }
          if (m_rank_calculate_std_dev) {
            message << " : " << std::setw(15) << getRankStdDev(i);
          }
          if (m_rank_calculate_maximum) {
            message << " : " << std::setw(15) << getRankMaximum(i)
                    << " : " << std::setw(maxRankStrLength) << getRankForMaximum(i);
          }
          if( m_calcImbalance ) {
            if( getRankMaximum(i) == 0.0 )
              message << " : 0";
            else
              message << " : " << 100.0 * (1.0 - (getRankAverage(i) /
                                                  getRankMaximum(i)));
          }
        }
      }

      if (message.str().size()) {
        if( oType == BaseInfoMapper::Dout ) {
          DOUT(true, header.str() + message.str());
        } else {
          std::ofstream fout;
          std::string filename = statsName +
            (nRanks != -1 ? "." + std::to_string(nRanks)   : "") +
            (rank   != -1 ? "." + std::to_string(rank)     : "") +
            (oType == Write_Separate ? "." + std::to_string(timeStep) : "");

          if( oType == Write_Append )
            fout.open(filename, std::ofstream::out | std::ofstream::app);
          else
            fout.open(filename, std::ofstream::out);

          fout << header.str() << message.str() << std::endl;
          fout.close();
        }
      }
    }
  }

  //______________________________________________________________________
  void reportNodeSummaryStats( const std::string statsName,
                               const std::string preamble,
                               const int rank,
                               const int nRanks,
                               const int node,
                               const int nNodes,
                               const int timeStep,
                               const double simTime,
                               const OutputTypeEnum oType,
                               bool calcImbalance )
  {
    unsigned int nStats = InfoMapper<E, T>::m_keys.size();

    if( nStats == 0 )
      return;

    // Get the max string length so to have descent formatting.
    std::ostringstream tmp;
    tmp << nRanks;

    unsigned int maxRankStrLength = std::max( (int) 4, (int) tmp.str().size() );
    unsigned int maxStatStrLength = std::string("Description").size();
    unsigned int maxUnitStrLength = 0;

    for (unsigned int i=0; i<nStats; ++i) {
      if (InfoMapper<E, T>::getRankValue(i) > 0) {

        if ( maxStatStrLength < InfoMapper<E, T>::getName(i).size() )
          maxStatStrLength = InfoMapper<E, T>::getName(i).size();

        if ( maxUnitStrLength < InfoMapper<E, T>::getUnits(i).size() )
          maxUnitStrLength = InfoMapper<E, T>::getUnits(i).size();
      }
    }

    // All ranks on a node.
    if( m_node_calculate_sum ||
        m_node_calculate_average || m_node_calculate_std_dev ||
        m_node_calculate_minimum || m_node_calculate_maximum ) {

      bool m_calcImbalance = calcImbalance &&
        m_rank_calculate_average && m_rank_calculate_maximum;

      std::ostringstream header;

      if( !preamble.empty() )
        header << preamble << std::endl;

      header << statsName << " summary stats for "
             << "time step " << timeStep
             << " at time="  << simTime
             << std::endl
             << "  " << std::left
             << std::setw(maxStatStrLength+2) << "Description"
             << std::setw(maxUnitStrLength+5) << "Units";
      if (m_node_calculate_sum) {
        std::ostringstream tmp;
        tmp << "Sum (" << nRanks << ")";
        header << std::setw(18) << tmp.str();
      }
      if (m_node_calculate_minimum) {
        header << std::setw(18) << "Minimum"
               << std::setw(maxRankStrLength+3) << "Rank";
      }
      if (m_node_calculate_average) {
        std::ostringstream tmp;
        tmp << "Average (" << nRanks << ")";
        header << std::setw(18) << tmp.str();
      }
      if (m_node_calculate_std_dev) {
        header << std::setw(18) << "Std. Dev.";
      }
      if (m_node_calculate_maximum) {
        header << std::setw(18) << "Maximum"
               << std::setw(maxRankStrLength+3) << "Rank";
      }
      if( m_calcImbalance ) {
        header << std::setw(12) << "100*(1-ave/max) '% load imbalance'";
      }

      header << std::endl;

      std::ostringstream message;

      for (unsigned int i=0; i<InfoMapper<E, T>::size(); ++i) {
        if( getNodeSum(i)     != 0.0 || getNodeAverage(i) != 0.0 ||
            getNodeMinimum(i) != 0.0 || getNodeMaximum(i) != 0.0 ) {
          if (message.str().size()) {
            message << std::endl;
          }

          message << "  " << std::left << std::setw(maxStatStrLength+2)
                  << InfoMapper<E, T>::getName(i) << "[" << std::setw(maxUnitStrLength)
                  << InfoMapper<E, T>::getUnits(i) << "]";

          if (m_node_calculate_sum) {
            message << " : " << std::setw(15) << getNodeSum(i);
          }
          if (m_node_calculate_minimum) {
            message << " : " << std::setw(15) << getNodeMinimum(i)
                    << " : " << std::setw(maxRankStrLength)  << getNodeForMinimum(i);
          }
          if (m_node_calculate_average) {
            message << " : " << std::setw(15) << getNodeAverage(i);
          }
          if (m_node_calculate_std_dev) {
            message << " : " << std::setw(15) << getNodeStdDev(i);
          }
          if (m_node_calculate_maximum) {
            message << " : " << std::setw(15) << getNodeMaximum(i)
                    << " : " << std::setw(maxRankStrLength)  << getNodeForMaximum(i);
          }
          if( m_calcImbalance ) {
            if( getNodeMaximum(i) == 0.0 )
              message << " : 0";
            else
              message << " : " << 100.0 * (1.0 - (getNodeAverage(i) /
                                                  getNodeMaximum(i)));
          }
        }
      }

      if (message.str().size()) {
        if( oType == BaseInfoMapper::Dout ) {
          DOUT(true, header.str() + message.str());
        } else {
          std::ofstream fout;
          std::string filename = statsName +
            (nNodes != -1 ? "." + std::to_string(nNodes)   : "") +
            (node   != -1 ? "." + std::to_string(node)     : "") +
            (nRanks != -1 ? "." + std::to_string(nRanks)   : "") +
            (rank   != -1 ? "." + std::to_string(rank)     : "") +
            (oType == Write_Separate ? "." + std::to_string(timeStep) : "");

          if( oType == Write_Append )
            fout.open(filename, std::ofstream::out | std::ofstream::app);
          else
            fout.open(filename, std::ofstream::out);

          fout << header.str() << message.str() << std::endl;
          fout.close();
        }
      }
    }
  }

  //______________________________________________________________________
  void reportIndividualStats( const std::string statsName,
                              const std::string preamble,
                              const int rank,
                              const int nRanks,
                              const int timeStep,
                              const double simTime,
                              const OutputTypeEnum oType )

  {
    unsigned int nStats = InfoMapper<E, T>::m_keys.size();

    if( nStats == 0 )
      return;

    // Get the max string length so to have descent formatting.
    std::ostringstream tmp;
    tmp << nRanks;

    unsigned int maxRankStrLength = tmp.str().size();
    unsigned int maxStatStrLength = 0;
    unsigned int maxUnitStrLength = 0;

    for (unsigned int i=0; i<nStats; ++i) {
      if (InfoMapper<E, T>::getRankValue(i) > 0) {

        if ( maxStatStrLength < InfoMapper<E, T>::getName(i).size() )
          maxStatStrLength = InfoMapper<E, T>::getName(i).size();

        if ( maxUnitStrLength < InfoMapper<E, T>::getUnits(i).size() )
          maxUnitStrLength = InfoMapper<E, T>::getUnits(i).size();
      }
    }

    std::ostringstream header;

    if( !preamble.empty() )
      header << preamble << std::endl;

    header << "--" << std::left
           << "Rank: " << std::setw(5) << rank << "  "
           << statsName << " stats for "
           << "time step " << timeStep
           << " at time="  << simTime
           << std::endl;

    std::ostringstream message;

    for (unsigned int i=0; i<nStats; ++i) {
      if (InfoMapper<E, T>::getRankValue(i) > 0) {
        if (message.str().size()) {
          message << std::endl;
        }
        message << "  " << std::left
                << "Rank: " << std::setw(maxRankStrLength+2) << rank
                << std::left << std::setw(maxStatStrLength+2) << InfoMapper<E, T>::getName(i)
                << "["   << std::setw(maxUnitStrLength) << InfoMapper<E, T>::getUnits(i) << "]"
                << " : " << std::setw(15) << InfoMapper<E, T>::getRankValue(i);
        if( InfoMapper<E, T>::getCount(i) )
          message << " ("  << std::setw( 4) << InfoMapper<E, T>::getCount(i) << ")";
      }
    }

    if (message.str().size()) {
      if( oType == BaseInfoMapper::Dout ) {
        DOUT(true, header.str() + message.str());
      } else {
        std::ofstream fout;
        std::string filename = statsName +
          (nRanks != -1 ? "." + std::to_string(nRanks)   : "") +
          (rank   != -1 ? "." + std::to_string(rank)     : "") +
          (oType == Write_Separate ? "." + std::to_string(timeStep) : "");

        if( oType == Write_Append )
          fout.open(filename, std::ofstream::out | std::ofstream::app);
        else
          fout.open(filename, std::ofstream::out);

        fout << header.str() << message.str() << std::endl;
        fout.close();
      }
    }
  }

protected:
  bool m_rank_calculate_sum    {false};
  bool m_rank_calculate_average{true};
  bool m_rank_calculate_minimum{false};
  bool m_rank_calculate_maximum{true};
  bool m_rank_calculate_std_dev{false};

  bool m_node_calculate_sum    {false};
  bool m_node_calculate_average{false};
  bool m_node_calculate_minimum{false};
  bool m_node_calculate_maximum{false};
  bool m_node_calculate_std_dev{false};

  std::vector< double >     m_rank_sum;     //     Sum over all ranks utilized
  std::vector< double >     m_rank_average; // Average over all ranks utilized
  std::vector< double_int > m_rank_minimum; // Minimum over all ranks utilized
  std::vector< double_int > m_rank_maximum; // Maximum over all ranks utilized
  std::vector< double >     m_rank_std_dev; // Std Dev over all ranks utilized

  std::vector< double >     m_node_sum;     //     Sum over all ranks on a single node
  std::vector< double >     m_node_average; // Average over all ranks on a single node
  std::vector< double_int > m_node_minimum; // Minimum over all ranks on a single node
  std::vector< double_int > m_node_maximum; // Maximum over all ranks on a single node
  std::vector< double >     m_node_std_dev; // Std Dev over all ranks on a single node
};

////////////////////////////////////////////////////////////////////////////////
// A vector of info mappers. Used on a per rank basis with indexes.
template<class E, class T> class VectorInfoMapper : public BaseInfoMapper
{
public:

  VectorInfoMapper()
  {
    clear();
  };

  virtual ~VectorInfoMapper() {};

  virtual void calculateSum    ( bool val ) { m_calculate_sum     = val; }
  virtual void calculateAverage( bool val ) { m_calculate_average = val; }
  virtual void calculateMinimum( bool val ) { m_calculate_minimum = val; }
  virtual void calculateMaximum( bool val ) { m_calculate_maximum = val; }
  virtual void calculateStdDev ( bool val ) { m_calculate_std_dev = val; }

  virtual bool calculateSum()     const { return m_calculate_sum; }
  virtual bool calculateAverage() const { return m_calculate_average; }
  virtual bool calculateMinimum() const { return m_calculate_minimum; }
  virtual bool calculateMaximum() const { return m_calculate_maximum; }
  virtual bool calculateStdDev () const { return m_calculate_std_dev; }

        InfoMapper<E, T>& operator[](unsigned int index)       { return m_vecInfoMapper[index]; };
  const InfoMapper<E, T>& operator[](unsigned int index) const { return m_vecInfoMapper[index]; };

  virtual void clear()
  {
    m_vecInfoMapper.clear();

    m_sum.clear();
    m_average.clear();
    m_minimum.clear();
    m_maximum.clear();
    m_std_dev.clear();
  };

  virtual void setIndexName( const std::string name )
  {
    m_indexName = name;
  }

  virtual size_t size() const
  {
    return m_vecInfoMapper.size();
  }

  virtual void resize( size_t n )
  {
    m_vecInfoMapper.resize(n);

    m_average.clear();
    m_minimum.clear();
    m_maximum.clear();
    m_std_dev.clear();
  };

  virtual void reset( const T val )
  {
    for( unsigned int i=0; i<m_vecInfoMapper.size(); ++i )
      m_vecInfoMapper[i].reset( val );
  };

  virtual void insert( const E key, const std::string name,
                       const std::string units )
  {
    for( unsigned int i=0; i<m_vecInfoMapper.size(); ++i )
      m_vecInfoMapper[i].insert( key, name, units );

    m_sum.push_back(0);
    m_average.push_back(0);
    m_minimum.push_back(double_int(0,-1));
    m_maximum.push_back(double_int(0,-1));
    m_std_dev.push_back(0);
  }

  // Get sum over all entries
  virtual double getSum( const E key )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    m_vecInfoMapper[0].validKey( key );

    return m_sum[ m_vecInfoMapper[0].m_keys[key] ];
  };

  virtual double getSum( const unsigned int index )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(index);

    return getSum( key );
  };

  virtual double getSum( const std::string name )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(name);

    return getSum( key );
  };

  // Get average over all entries
  virtual double getAverage( const E key )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    m_vecInfoMapper[0].validKey( key );

    return m_average[ m_vecInfoMapper[0].m_keys[key] ];
  };

  virtual double getAverage( const unsigned int index )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(index);

    return getAverage( key );
  };

  virtual double getAverage( const std::string name )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(name);

    return getAverage( key );
  };

  // Get minium over all entries
  virtual double getMinimum( const E key )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    m_vecInfoMapper[0].validKey( key );

    return m_minimum[ m_vecInfoMapper[0].m_keys[key] ].val;
  };

  virtual double getMinimum( const unsigned int index )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(index);

    return getMinimum( key );
  };

  virtual double getMinimum( const std::string name )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(name);

    return getMinimum( key );
  };

  // Get index for the minimum entry
  virtual unsigned int getIndexForMinimum( const E key )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    m_vecInfoMapper[0].validKey( key );

    return m_minimum[ m_vecInfoMapper[0].m_keys[key] ].rank;
  };

  virtual unsigned int getIndexForMinimum( const unsigned int index )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(index);

    return getIndexForMinimum( key );
  };

  virtual unsigned int getIndexForMinimum( const std::string name )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(name);

    return getIndexForMinimum( key );
  };

  // Get maximum over all entries
  virtual double getMaximum( const E key )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    m_vecInfoMapper[0].validKey( key );

    return m_maximum[ m_vecInfoMapper[0].m_keys[key] ].val;
  };

  virtual double getMaximum( const unsigned int index )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(index);

    return getMaximum( key );
  };

  virtual double getMaximum( const std::string name )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(name);

    return getMaximum( key );
  };

  // Get index for the maximum entry
  virtual unsigned int getIndexForMaximum( const E key )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    m_vecInfoMapper[0].validKey( key );

    return m_maximum[ m_vecInfoMapper[0].m_keys[key] ].rank;
  };

  virtual unsigned int getIndexForMaximum( const unsigned int index )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(index);

    return getIndexForMaximum( key );
  };

  virtual unsigned int getIndexForMaximum( const std::string name )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(name);

    return getIndexForMaximum( key );
  };

  // Get std dev over all entries
  virtual double getStdDev( const E key )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    m_vecInfoMapper[0].validKey( key );

    return m_std_dev[ m_vecInfoMapper[0].m_keys[key] ];
  };

  virtual double getStdDev( const unsigned int index )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(index);

    return getStdDev( key );
  };

  virtual double getStdDev( const std::string name )
  {
    if( m_vecInfoMapper.size() == 0 )
      return 0;

    const E key = m_vecInfoMapper[0].getKey(name);

    return getStdDev( key );
  };

  // Reduce
  virtual void reduce( bool skipFirst )
  {
    if( m_vecInfoMapper.size() == 0 ) {
      return;
    }

    m_skipFirst = skipFirst;

    unsigned int nStats = m_vecInfoMapper[0].m_keys.size();

    if (nStats == 0) {
      return;
    }

    m_sum.resize(nStats);
    m_average.resize(nStats);
    m_minimum.resize(nStats);
    m_maximum.resize(nStats);
    m_std_dev.resize(nStats);

    for (size_t i = 0; i < nStats; ++i) {
      m_sum[i] = 0;
    }

    if (m_vecInfoMapper.size() > 1) {

      // Calculate each stat.
      for (size_t i = 0; i < nStats; ++i) {

        m_minimum[i] = double_int( DBL_MAX, -1);
        m_maximum[i] = double_int(-DBL_MAX, -1);

        // Calculate across all entries.
        for (size_t j = int(m_skipFirst); j < m_vecInfoMapper.size(); ++j) {

          double val;
          if( m_vecInfoMapper[j].m_counts[i] )
            val = m_vecInfoMapper[j].m_values[i] / m_vecInfoMapper[j].m_counts[i];
          else
            val = m_vecInfoMapper[j].m_values[i];

          // Sum across all entires.
          if( m_calculate_sum || m_calculate_average || m_calculate_std_dev )
            m_sum[i] += val;

          // Min across all entries.
          if( m_calculate_minimum ) {
            if( val < m_minimum[i].val )
               m_minimum[i] = double_int(val, j);
          }

          // Max across all entries.
          if( m_calculate_maximum ) {
            if( m_maximum[i].val < val )
               m_maximum[i] = double_int(val, j);
          }
        }

        // Sums across all entries.
        if( m_calculate_average || m_calculate_std_dev )
        {
          // Calculate the average.
          m_average[i] = m_sum[i] / double(m_vecInfoMapper.size() - int(m_skipFirst));

          if( m_calculate_std_dev )
          {
            //  Calculate the sum of squared differences
            double sum = 0;
            for (size_t j = int(m_skipFirst); j < m_vecInfoMapper.size(); ++j) {

              double val;
              if( m_vecInfoMapper[j].m_counts[i] )
                val = m_vecInfoMapper[j].m_values[i] / m_vecInfoMapper[j].m_counts[i];
              else
                val = m_vecInfoMapper[j].m_values[i];

              sum = (val - m_average[i]) * (val - m_average[i]);
            }

            if( m_vecInfoMapper.size() - int(m_skipFirst) - 1 > 0 )
              m_std_dev[i] = std::sqrt(sum / double(m_vecInfoMapper.size() - int(m_skipFirst) - 1) );
            else
              m_std_dev[i] = 0;
          }
        }
      }
    }

    // Single entry so just copy the values.
    else {
      for (size_t i = 0; i < nStats; ++i) {
        double val;

        if(m_vecInfoMapper[0].m_counts[i] )
          val = m_vecInfoMapper[0].m_values[i] / m_vecInfoMapper[0].m_counts[i];
        else
          val = m_vecInfoMapper[0].m_values[i];

        m_sum[i] = val;
        m_average[i] = val;
        m_minimum[i] = double_int(val, 0);
        m_maximum[i] = double_int(val, 0);
        m_std_dev[i] = 0;
      }
    }
  };

  //______________________________________________________________________
  //
  void reportSummaryStats( const std::string statsName,
                           const std::string preamble,
                           const int rank,
                           const int nRanks,
                           const int timeStep,
                           const double simTime,
                           const OutputTypeEnum oType,
                           bool calcImbalance )
  {
    if( m_vecInfoMapper.size() == 0 )
      return;

    unsigned int nStats = m_vecInfoMapper[0].size();

    if( nStats == 0 )
      return;

    // Get the max string length so to have descent formatting.
    std::ostringstream tmp;
    tmp << nRanks;

    unsigned int maxRankStrLength  = tmp.str().size();
    unsigned int maxIndexStrLength = m_indexName.size();
    unsigned int maxStatStrLength  = std::string("Description").size();
    unsigned int maxUnitStrLength  = 0;

    for (unsigned int j=0; j<m_vecInfoMapper.size(); ++j) {
      for (unsigned int i=0; i<m_vecInfoMapper[j].size(); ++i) {
        if (m_vecInfoMapper[j][i] > 0) {

          std::ostringstream tmp;
          tmp << j;

          if ( maxIndexStrLength < tmp.str().size() )
            maxIndexStrLength = tmp.str().size();

          if ( maxStatStrLength < m_vecInfoMapper[j].getName(i).size() )
            maxStatStrLength = m_vecInfoMapper[j].getName(i).size();

          if ( maxUnitStrLength < m_vecInfoMapper[j].getUnits(i).size() )
            maxUnitStrLength = m_vecInfoMapper[j].getUnits(i).size();
        }
      }
    }

    bool m_calcImbalance = calcImbalance &&
      m_calculate_average && m_calculate_maximum;

    std::ostringstream header;

    if( !preamble.empty() )
      header << preamble << std::endl;

    header << "Rank: " << std::setw(maxRankStrLength) << rank << "  "
           << statsName << " summary stats for "
           << "time step " << timeStep
           << " at time="  << simTime
           << std::endl
           << "  " << std::left
           << std::setw(maxStatStrLength+2) << "Description"
           << std::setw(maxUnitStrLength+5) << "Units";
    if (m_calculate_sum) {
      std::ostringstream tmp;
      tmp << "Sum (" << m_vecInfoMapper.size() << ")";
      header << std::setw(18) << tmp.str();
    }
    if (m_calculate_minimum) {
      header << std::setw(18) << "Minimum"
             << std::setw(maxIndexStrLength+3) << m_indexName;
    }
    if (m_calculate_average) {
      std::ostringstream tmp;
      tmp << "Average (" << m_vecInfoMapper.size() << ")";
      header << std::setw(18) << tmp.str();
    }
    if (m_calculate_std_dev) {
      header << std::setw(18) << "Std. Dev.";
    }
    if (m_calculate_maximum) {
      header << std::setw(18) << "Maximum"
             << std::setw(maxIndexStrLength+3) << m_indexName;
    }
    if( m_calcImbalance ) {
      header << std::setw(12) << "100*(1-ave/max) '% load imbalance'";
    }

    header << std::endl;

    std::ostringstream message;

    for (unsigned int i=0; i<nStats; ++i) {
      if( getSum(i)     != 0.0 || getAverage(i) != 0.0 ||
          getMinimum(i) != 0.0 || getMaximum(i) != 0.0 )
        {
          if (message.str().size()) {
            message << std::endl;
          }

          message << "  " << std::left << std::setw(maxStatStrLength+2)
                  << m_vecInfoMapper[0].getName(i) << "[" << std::setw(maxUnitStrLength)
                  << m_vecInfoMapper[0].getUnits(i) << "]";

          if (m_calculate_sum) {
            message << " : " << std::setw(15) << getSum(i);
          }

          if (m_calculate_minimum) {
            message << " : " << std::setw(15) << getMinimum(i)
                    << " : " << std::setw(maxIndexStrLength)  << getIndexForMinimum(i);
          }

          if (m_calculate_average) {
            message << " : " << std::setw(15) << getAverage(i);
          }

          if (m_calculate_std_dev) {
            message << " : " << std::setw(15) << getStdDev(i);
          }

          if (m_calculate_maximum) {
            message << " : " << std::setw(15) << getMaximum(i)
                    << " : " << std::setw(maxIndexStrLength)  << getIndexForMaximum(i);
          }

          if( m_calcImbalance ) {
            if( getMaximum(i) == 0.0 )
              message << " : 0";
            else
              message << " : " << 100.0 * (1.0 - (getAverage(i) / getMaximum(i)));
          }
        }
    }

    if( message.str().size() ) {
      if( oType == BaseInfoMapper::Dout ) {
        DOUT(true, header.str() + message.str());
      } else {
        std::ofstream fout;
        std::string filename = statsName +
          (nRanks != -1 ? "." + std::to_string(nRanks)   : "") +
          (rank   != -1 ? "." + std::to_string(rank)     : "") +
          (oType == Write_Separate ? "." + std::to_string(timeStep) : "");

        if( oType == Write_Append )
          fout.open(filename, std::ofstream::out | std::ofstream::app);
        else
          fout.open(filename, std::ofstream::out);

        fout << header.str() << message.str() << std::endl;
        fout.close();
      }
    }
  }

  //______________________________________________________________________
  //
  void reportIndividualStats( const std::string statsName,
                              const std::string preamble,
                              const int rank,
                              const int nRanks,
                              const int timeStep,
                              const double simTime,
                              const OutputTypeEnum oType )
  {
    if( m_vecInfoMapper.size() == 0 )
      return;

    unsigned int nStats = m_vecInfoMapper[0].size();

    if( nStats == 0 )
      return;

    // Get the max string length so to have descent formatting.
    std::ostringstream tmp;
    tmp << nRanks;

    unsigned int maxRankStrLength = tmp.str().size();
    unsigned int maxStatStrLength = 0;
    unsigned int maxUnitStrLength = 0;

    for (unsigned int j=0; j<m_vecInfoMapper.size(); ++j) {
      for (unsigned int i=0; i<m_vecInfoMapper[j].size(); ++i) {
        if (m_vecInfoMapper[j][i] > 0) {

          if ( maxStatStrLength < m_vecInfoMapper[j].getName(i).size() )
            maxStatStrLength = m_vecInfoMapper[j].getName(i).size();

          if ( maxUnitStrLength < m_vecInfoMapper[j].getUnits(i).size() )
            maxUnitStrLength = m_vecInfoMapper[j].getUnits(i).size();
        }
      }
    }

    std::ostringstream header;

    if( !preamble.empty() )
      header << preamble << std::endl;

    header << "--" << std::left
           << "Rank: " << std::setw(maxRankStrLength) << rank << "  "
           << statsName << " stats for "
           << "time step " << timeStep
           << " at time="  << simTime
           << std::endl;

    std::ostringstream message;

    for (unsigned int j=0; j<m_vecInfoMapper.size(); ++j) {
      for (unsigned int i=0; i<m_vecInfoMapper[j].size(); ++i) {
        if (m_vecInfoMapper[j][i] > 0) {
          if (message.str().size()) {
            message << std::endl;
          }
          message << "  " << std::left
                  << "Rank: " << std::setw(maxRankStrLength+2) << rank
                  << m_indexName << ": " << std::setw(4) << j
                  << std::left << std::setw(maxStatStrLength+2) << m_vecInfoMapper[j].getName(i)
                  << "["   << std::setw(maxUnitStrLength) << m_vecInfoMapper[j].getUnits(i) << "]"
                  << " : " << std::setw(15) << m_vecInfoMapper[j].getRankValue(i);
          if( m_vecInfoMapper[j].getCount(i) )
            message << " ("  << std::setw( 4) << m_vecInfoMapper[j].getCount(i) << ")";
        }
      }
    }

    if (message.str().size()) {
      if( oType == BaseInfoMapper::Dout ) {
        DOUT(true, header.str() + message.str());
      } else {
        std::ofstream fout;
        std::string filename = statsName +
          (nRanks != -1 ? "." + std::to_string(nRanks)   : "") +
          (rank   != -1 ? "." + std::to_string(rank)     : "") +
          (oType == Write_Separate ? "." + std::to_string(timeStep) : "");

        if( oType == Write_Append )
          fout.open(filename, std::ofstream::out | std::ofstream::app);
        else
          fout.open(filename, std::ofstream::out);

        fout << header.str() << message.str() << std::endl;
        fout.close();
      }
    }
  }

protected:
  bool m_skipFirst;
  std::string m_indexName;

  std::vector< InfoMapper<E, T> > m_vecInfoMapper;

  bool m_calculate_sum    {true};
  bool m_calculate_average{true};
  bool m_calculate_minimum{false};
  bool m_calculate_maximum{true};
  bool m_calculate_std_dev{false};

  std::vector< double >     m_sum;      // Sum     over all entries
  std::vector< double >     m_average;  // Average over all entries
  std::vector< double_int > m_minimum;  // Minimum over all entries
  std::vector< double_int > m_maximum;  // Maximum over all entries
  std::vector< double >     m_std_dev;  // Standard deviation over all entries
};

////////////////////////////////////////////////////////////////////////////////
// A mapped info mapper. Used on a per rank basis with different keys.
template<class KEY, class E, class T> class MapInfoMapper : public BaseInfoMapper
{
public:

  MapInfoMapper()
  {
    clear();
  };

  virtual ~MapInfoMapper() {};

  virtual void calculateSum    ( bool val ) { m_calculate_sum     = val; }
  virtual void calculateAverage( bool val ) { m_calculate_average = val; }
  virtual void calculateMinimum( bool val ) { m_calculate_minimum = val; }
  virtual void calculateMaximum( bool val ) { m_calculate_maximum = val; }
  virtual void calculateStdDev ( bool val ) { m_calculate_std_dev = val; }

  virtual bool calculateSum()     const { return m_calculate_sum; }
  virtual bool calculateAverage() const { return m_calculate_average; }
  virtual bool calculateMinimum() const { return m_calculate_minimum; }
  virtual bool calculateMaximum() const { return m_calculate_maximum; }
  virtual bool calculateStdDev () const { return m_calculate_std_dev; }

  InfoMapper<E, T>& operator[](KEY key) {

    // If the key does not exist insert the keys, names, and units.
    if( m_mapInfoMapper.find(key) == m_mapInfoMapper.end() ) {
      for ( unsigned int i=0; i<m_keys.size(); ++i ) {
        m_mapInfoMapper[key].insert( m_keys[i], m_names[i], m_units[i] );
      }
    }

    return m_mapInfoMapper[key];
  };

  const InfoMapper<E, T>& operator[](KEY key) const {
    return m_mapInfoMapper[key];
  };

  virtual void clear()
  {
    m_mapInfoMapper.clear();

    m_keys.clear();
    m_names.clear();
    m_units.clear();

    m_sum.clear();
    m_average.clear();
    m_minimum.clear();
    m_maximum.clear();
    m_std_dev.clear();
  };

  virtual size_t size() const
  {
    return m_mapInfoMapper.size();
  }

  virtual void setKeyName( const std::string name )
  {
    m_keyName = name;
  };

  // Get the key via an index.
  virtual KEY getKey( const unsigned int index )
  {
    if( index < 0 || m_mapInfoMapper.size() <= index ) {
      std::stringstream msg;
      msg << "Requesting an undefined key index (" << index << ") ";
      throw Uintah::InternalError( msg.str(), __FUNCTION__, __LINE__);
    }

    typename std::map< KEY, InfoMapper<E, T> >::iterator it =
      m_mapInfoMapper.begin();

    for( unsigned int i=0; i<index; ++i )
      ++it;

    return it->first;
  };

  virtual void reset( const T val )
  {
    for ( auto & var : m_mapInfoMapper )
      var.second.reset( val );
  };

  virtual void insert( const E key, const std::string name,
                       const std::string units )
  {
    m_keys.push_back(key);
    m_names.push_back(name);
    m_units.push_back(units);

    m_sum.push_back(0);
    m_average.push_back(0);
    m_minimum.push_back( std::pair<double, KEY>(0, KEY()) );
    m_maximum.push_back( std::pair<double, KEY>(0, KEY()) );
    m_std_dev.push_back(0);
  }

  // Get sum over all entries
  virtual double getSum( const E key )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    m_mapInfoMapper.begin()->second.validKey( key );

    return m_sum[ m_mapInfoMapper.begin()->second.m_keys[key] ];
  };

  virtual double getSum( const unsigned int index )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(index);

    return getSum( key );
  };

  virtual double getSum( const std::string name )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(name);

    return getSum( key );
  };

  // Get average over all entries
  virtual double getAverage( const E key )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    m_mapInfoMapper.begin()->second.validKey( key );

    return m_average[ m_mapInfoMapper.begin()->second.m_keys[key] ];
  };

  virtual double getAverage( const unsigned int index )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(index);

    return getAverage( key );
  };

  virtual double getAverage( const std::string name )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(name);

    return getAverage( key );
  };

  // Get minium over all entries
  virtual double getMinimum( const E key )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    m_mapInfoMapper.begin()->second.validKey( key );

    return m_minimum[ m_mapInfoMapper.begin()->second.m_keys[key] ].first;
  };

  virtual double getMinimum( const unsigned int index )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(index);

    return getMinimum( key );
  };

  virtual double getMinimum( const std::string name )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(name);

    return getMinimum( key );
  };

  // Get index for the minimum entry
  virtual KEY getIndexForMinimum( const E key )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    m_mapInfoMapper.begin()->second.validKey( key );

    return m_minimum[ m_mapInfoMapper.begin()->second.m_keys[key] ].second;
  };

  virtual KEY getIndexForMinimum( const unsigned int index )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(index);

    return getIndexForMinimum( key );
  };

  virtual KEY getIndexForMinimum( const std::string name )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(name);

    return getIndexForMinimum( key );
  };

  // Get maximum over all entries
  virtual double getMaximum( const E key )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    m_mapInfoMapper.begin()->second.validKey( key );

    return m_maximum[ m_mapInfoMapper.begin()->second.m_keys[key] ].first;
  };

  virtual double getMaximum( const unsigned int index )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(index);

    return getMaximum( key );
  };

  virtual double getMaximum( const std::string name )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(name);

    return getMaximum( key );
  };

  // Get index for the maximum entry
  virtual KEY getIndexForMaximum( const E key )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    m_mapInfoMapper.begin()->second.validKey( key );

    return m_maximum[ m_mapInfoMapper.begin()->second.m_keys[key] ].second;
  };

  virtual KEY getIndexForMaximum( const unsigned int index )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(index);

    return getIndexForMaximum( key );
  };

  virtual KEY getIndexForMaximum( const std::string name )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(name);

    return getIndexForMaximum( key );
  };

  // Get std dev over all entries
  virtual double getStdDev( const E key )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    m_mapInfoMapper.begin()->second.validKey( key );

    return m_std_dev[ m_mapInfoMapper.begin()->second.m_keys[key] ];
  };

  virtual double getStdDev( const unsigned int index )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(index);

    return getStdDev( key );
  };

  virtual double getStdDev( const std::string name )
  {
    if( m_mapInfoMapper.size() == 0 )
      return 0;

    const E key = m_mapInfoMapper.begin()->second.getKey(name);

    return getStdDev( key );
  };

  // Reduce
  virtual void reduce( bool skipFirst )
  {
    if( m_mapInfoMapper.size() == 0 )
      return;

    m_skipFirst = skipFirst;

    unsigned int nStats = m_mapInfoMapper.begin()->second.m_keys.size();

    if (nStats == 0) {
      return;
    }

    m_sum.resize(nStats);
    m_average.resize(nStats);
    m_minimum.resize(nStats);
    m_maximum.resize(nStats);
    m_std_dev.resize(nStats);

    for (size_t i = 0; i < nStats; ++i) {
      m_sum[i] = 0;
    }

    if (m_mapInfoMapper.size() > 1) {

      // Calculate each stat.
      for (size_t i = 0; i < nStats; ++i) {

        m_minimum[i] = std::pair<double, KEY>(  DBL_MAX, KEY());
        m_maximum[i] = std::pair<double, KEY>( -DBL_MAX, KEY());

        int cc = 0;

        // Calculate across all entries.
        for ( auto & var : m_mapInfoMapper ) {

          if( m_skipFirst && cc++ == 0 )
            continue;

          double val;
          if( var.second.m_counts[i] )
            val = var.second.m_values[i] / var.second.m_counts[i];
          else
            val = var.second.m_values[i];

          // Sum across all entires.
          if( m_calculate_sum || m_calculate_average || m_calculate_std_dev )
            m_sum[i] += val;

          // Min across all entries.
          if( m_calculate_minimum ) {
            if( val < m_minimum[i].first )
              m_minimum[i] = std::pair<double, KEY>(val, var.first);
          }

          // Max across all entries.
          if( m_calculate_maximum ) {
            if( m_maximum[i].first < val )
              m_maximum[i] = std::pair<double, KEY>(val, var.first);
          }
        }

        // Sums across all entries.
        if( m_calculate_average || m_calculate_std_dev )
        {
          // Calculate the average.
          m_average[i] = m_sum[i] / double(m_mapInfoMapper.size() - int(m_skipFirst));

          if( m_calculate_std_dev )
          {
            //  Calculate the sum of squared differences
            int cc = 0;
            double sum = 0;
            for ( auto & var : m_mapInfoMapper ) {

              if( m_skipFirst && cc++ == 0 )
                continue;

              double val;
              if( var.second.m_counts[i] )
                val = var.second.m_values[i] / var.second.m_counts[i];
              else
                val = var.second.m_values[i];

              sum = (val - m_average[i]) * (val - m_average[i]);
            }

            if( m_mapInfoMapper.size() - int(m_skipFirst) - 1 > 0 )
              m_std_dev[i] = std::sqrt(sum / double(m_mapInfoMapper.size() - int(m_skipFirst) - 1) );
            else
              m_std_dev[i] = 0;
          }
        }
      }
    }

    // Single entry so just copy the values.
    else {
      for (size_t i = 0; i < nStats; ++i) {
        double val;

        if(m_mapInfoMapper.begin()->second.m_counts[i] )
          val = m_mapInfoMapper.begin()->second.m_values[i] /
            m_mapInfoMapper.begin()->second.m_counts[i];
        else
          val = m_mapInfoMapper.begin()->second.m_values[i];

        KEY key = m_mapInfoMapper.begin()->first;

        m_sum    [i] = val;
        m_average[i] = val;
        m_minimum[i] = std::pair<double, KEY>(val, key);
        m_maximum[i] = std::pair<double, KEY>(val, key);
        m_std_dev[i] = 0;
      }
    }
  };

  //______________________________________________________________________
  //
  void reportSummaryStats( const std::string statsName,
                           const std::string preamble,
                           const int rank,
                           const int nRanks,
                           const int timeStep,
                           const double simTime,
                           const OutputTypeEnum oType,
                           bool calcImbalance )
  {
    if( m_mapInfoMapper.size() == 0 )
      return;

    unsigned int nStats = m_mapInfoMapper.begin()->second.size();

    if( nStats == 0 )
      return;

    // Get the max string length so to have descent formatting.
    std::ostringstream tmp;
    tmp << nRanks;

    unsigned int maxRankStrLength  = tmp.str().size();
    unsigned int maxKeyStrLength   = m_keyName.size();
    unsigned int maxStatStrLength  = std::string("Description").size();
    unsigned int maxUnitStrLength  = std::string("Units").size();

    for ( auto & var : m_mapInfoMapper ) {
      for (unsigned int i=0; i<var.second.size(); ++i) {
        if (var.second[i] > 0) {

          std::ostringstream tmp;
          tmp << var.first;

          if ( maxKeyStrLength < tmp.str().size() )
            maxKeyStrLength = tmp.str().size();

          if ( maxStatStrLength < var.second.getName(i).size() )
            maxStatStrLength = var.second.getName(i).size();

          if ( maxUnitStrLength < var.second.getUnits(i).size() )
            maxUnitStrLength = var.second.getUnits(i).size();
        }
      }
    }

    bool m_calcImbalance = calcImbalance &&
      m_calculate_average && m_calculate_maximum;

    std::ostringstream header;

    if( !preamble.empty() )
      header << preamble << std::endl;

    header << "Rank: " << std::setw(maxRankStrLength) << rank << "  "
           << statsName << " summary stats for "
           << "time step " << timeStep
           << " at time="  << simTime
           << std::endl
           << "  " << std::left
           << std::setw(maxStatStrLength+2) << "Description"
           << std::setw(maxUnitStrLength+5) << "Units";
    if (m_calculate_sum) {
      std::ostringstream tmp;
      tmp << "Sum (" << m_mapInfoMapper.size() << ")";
      header << std::setw(18) << tmp.str();
    }
    if (m_calculate_minimum) {
      header << std::setw(18) << "Minimum"
             << std::setw(maxKeyStrLength+3) << m_keyName;
    }
    if (m_calculate_average) {
      std::ostringstream tmp;
      tmp << "Average (" << m_mapInfoMapper.size() << ")";
      header << std::setw(18) << tmp.str();
    }
    if (m_calculate_std_dev) {
      header << std::setw(18) << "Std. Dev.";
    }
    if (m_calculate_maximum) {
      header << std::setw(18) << "Maximum"
             << std::setw(maxKeyStrLength+3) << m_keyName;
    }
    if( m_calcImbalance ) {
      header << std::setw(12) << "100*(1-ave/max) '% load imbalance'";
    }

    header << std::endl;

    std::ostringstream message;

    for (unsigned int i=0; i<nStats; ++i) {
      if( getSum(i)     != 0.0 || getAverage(i) != 0.0 ||
          getMinimum(i) != 0.0 || getMaximum(i) != 0.0 )
        {
          if (message.str().size()) {
            message << std::endl;
          }

          message << "  " << std::left << std::setw(maxStatStrLength+2)
                  << m_mapInfoMapper.begin()->second.getName(i) << "[" << std::setw(maxUnitStrLength)
                  << m_mapInfoMapper.begin()->second.getUnits(i) << "]";

          if (m_calculate_sum) {
            message << " : " << std::setw(15) << getSum(i);
          }

          if (m_calculate_minimum) {
            message << " : " << std::setw(15) << getMinimum(i)
                    << " : " << std::setw(maxKeyStrLength) << getIndexForMinimum(i);
          }

          if (m_calculate_average) {
            message << " : " << std::setw(15) << getAverage(i);
          }

          if (m_calculate_std_dev) {
            message << " : " << std::setw(15) << getStdDev(i);
          }

          if (m_calculate_maximum) {
            message << " : " << std::setw(15) << getMaximum(i)
                    << " : " << std::setw(maxKeyStrLength) << getIndexForMaximum(i);
          }

          if( m_calcImbalance ) {
            if( getMaximum(i) == 0.0 )
              message << " : 0";
            else
              message << " : " << 100.0 * (1.0 - (getAverage(i) / getMaximum(i)));
          }
        }
    }

    if( message.str().size() ) {
      if( oType == BaseInfoMapper::Dout ) {
        DOUT(true, header.str() + message.str());
      } else {
        std::ofstream fout;
        std::string filename = statsName +
          (nRanks != -1 ? "." + std::to_string(nRanks)   : "") +
          (rank   != -1 ? "." + std::to_string(rank)     : "") +
          (oType == Write_Separate ? "." + std::to_string(timeStep) : "");

        if( oType == Write_Append )
          fout.open(filename, std::ofstream::out | std::ofstream::app);
        else
          fout.open(filename, std::ofstream::out);

        fout << header.str() << message.str() << std::endl;
        fout.close();
      }
    }
  }

  //______________________________________________________________________
  //
  void reportIndividualStats( const std::string statsName,
                              const std::string preamble,
                              const int rank,
                              const int nRanks,
                              const int timeStep,
                              const double simTime,
                              const OutputTypeEnum oType )
  {
    if( m_mapInfoMapper.size() == 0 )
      return;

    unsigned int nStats = m_mapInfoMapper.begin()->second.size();

    if( nStats == 0 )
      return;

    // Get the max string length so to have descent formatting.
    std::ostringstream tmp;
    tmp << nRanks;

    unsigned int maxRankStrLength = tmp.str().size();
    unsigned int maxKeyStrLength = 0;
    unsigned int maxStatStrLength = 0;
    unsigned int maxUnitStrLength = 0;

    for ( auto & var : m_mapInfoMapper ) {
      for (unsigned int i=0; i<var.second.size(); ++i) {
        if (var.second[i] > 0) {

          std::ostringstream tmp;
          tmp << var.first;

          if ( maxKeyStrLength < tmp.str().size() )
            maxKeyStrLength = tmp.str().size();

          if ( maxStatStrLength < var.second.getName(i).size() )
            maxStatStrLength = var.second.getName(i).size();

          if ( maxUnitStrLength < var.second.getUnits(i).size() )
            maxUnitStrLength = var.second.getUnits(i).size();
        }
      }
    }

    std::ostringstream header;

    if( !preamble.empty() )
      header << preamble << std::endl;

    header << "--" << std::left
           << "Rank: " << std::setw(maxRankStrLength) << rank << "  "
           << statsName << " stats for "
           << "time step " << timeStep
           << " at time="  << simTime
           << std::endl;

    std::ostringstream message;

    for ( auto & var : m_mapInfoMapper ) {
      for (unsigned int i=0; i<var.second.size(); ++i) {
        if (var.second[i] > 0) {
          if (message.str().size()) {
            message << std::endl;
          }
          message << "  " << std::left
                  << "Rank: " << std::setw(maxRankStrLength+2) << rank
                  << m_keyName << ": "
                  << std::setw(maxKeyStrLength+2) << var.first
                  << std::setw(maxStatStrLength+2) << var.second.getName(i)
                  << "["   << std::setw(maxUnitStrLength) << var.second.getUnits(i) << "]"
                  << " : " << std::setw(15) << var.second.getRankValue(i);
          if( var.second.getCount(i) )
            message << " ("  << std::setw(4) << var.second.getCount(i) << ")";
        }
      }
    }

    if (message.str().size()) {
      if( oType == BaseInfoMapper::Dout ) {
        DOUT(true, header.str() + message.str());
      } else {
        std::ofstream fout;
        std::string filename = statsName +
          (nRanks != -1 ? "." + std::to_string(nRanks)   : "") +
          (rank   != -1 ? "." + std::to_string(rank)     : "") +
          (oType == Write_Separate ? "." + std::to_string(timeStep) : "");

        if( oType == Write_Append )
          fout.open(filename, std::ofstream::out | std::ofstream::app);
        else
          fout.open(filename, std::ofstream::out);

        fout << header.str() << message.str() << std::endl;
        fout.close();
      }
    }
  }

protected:
  bool m_skipFirst;
  std::string m_keyName;

  // For insterting the mapped values before the mapped key is created.
  std::vector< E >           m_keys;
  std::vector< std::string > m_names;
  std::vector< std::string > m_units;

  std::map< KEY, InfoMapper<E, T> > m_mapInfoMapper;

  bool m_calculate_sum    {true};
  bool m_calculate_average{true};
  bool m_calculate_minimum{false};
  bool m_calculate_maximum{true};
  bool m_calculate_std_dev{false};

  std::vector< double >     m_sum;      // Sum over all entries
  std::vector< double >     m_average;  // Average over all entries
  std::vector< std::pair<double, KEY> > m_minimum;  // Minimum over all entries
  std::vector< std::pair<double, KEY> > m_maximum;  // Maximum over all
  std::vector< double >     m_std_dev;  // Standard deviation over all entries
};

} // End namespace Uintah

#endif // CORE_UTIL_INFOMAPPER_H
