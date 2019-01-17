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
#include <Core/Util/DOUT.hpp>

#include <sci_defs/visit_defs.h>

#include <cfloat>
#include <cmath>
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
  template<class e, class t>
    friend class VectorInfoMapper;

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


template<class E, class T> class ReductionInfoMapper : public InfoMapper<E, T>
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

  virtual void calculateAverage( bool val ) { calculate_average = val; }
  virtual void calculateMinimum( bool val ) { calculate_minimum = val; }
  virtual void calculateMaximum( bool val ) { calculate_maximum = val; }
  virtual void calculateStdDev ( bool val ) { calculate_std_dev = val; }
  
  virtual bool calculateAverage() { return calculate_average; }
  virtual bool calculateMinimum() { return calculate_minimum; }
  virtual bool calculateMaximum() { return calculate_maximum; }
  virtual bool calculateStdDev () { return calculate_std_dev; }
  
  virtual void clear()
  {
    InfoMapper<E, T>::clear();

    m_rank_average.clear();
    m_rank_minimum.clear();
    m_rank_maximum.clear();
    m_rank_std_dev.clear();
  };

  virtual void insert( const E key, const std::string name,
                       const std::string units )
  {
    InfoMapper<E, T>::insert( key, name, units );

    m_rank_average.push_back(0);
    m_rank_minimum.push_back(double_int(0,-1));
    m_rank_maximum.push_back(double_int(0,-1));
    m_rank_std_dev.push_back(0);
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
      m_rank_minimum.resize(nStats);
      m_rank_maximum.resize(nStats);
      m_rank_std_dev.resize(nStats);

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
        toReduceStdDev[i] = val;
      }

      // Reduction across each node.
#ifdef HAVE_VISIT
      Uintah::MPI::Allreduce(&toReduce[0], &m_node_sum[0], nStats,
                             MPI_DOUBLE, MPI_SUM, myWorld->getNodeComm());

      for (size_t i = 0; i < nStats; ++i) {
        m_node_average[i] = m_node_sum[i] / myWorld->myNode_nRanks();
      }      
#endif
      
      // Sum reductions across all ranks.
      if( calculate_average || calculate_std_dev )
      {
        if (allReduce || calculate_std_dev ) {
          Uintah::MPI::Allreduce(&toReduce[0], &m_rank_average[0], nStats,
                                 MPI_DOUBLE, MPI_SUM, myWorld->getComm());
        }
        else {
          Uintah::MPI::Reduce(&toReduce[0], &m_rank_average[0], nStats,
                              MPI_DOUBLE, MPI_SUM, 0, myWorld->getComm());      
        }

        // Calculate the averages.
        for (size_t i = 0; i < nStats; ++i) {
          m_rank_average[i] /= myWorld->nRanks();
        }
        
        if( calculate_std_dev )
        {
          //  Calculate the squared differences
          for (size_t i = 0; i < nStats; ++i) {
            double val = toReduceStdDev[i] - m_rank_average[i];
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
            m_rank_std_dev[i] = std::sqrt(m_rank_std_dev[i] / (myWorld->nRanks()-1) );  
          }
        }
      }

      // Min reductions across all ranks.
      if( calculate_minimum )
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
      if( calculate_maximum )
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
    }

    // Single rank so just copy the values.
    else {
      m_node_sum.resize(nStats);
      m_node_average.resize(nStats);

      m_rank_average.resize(nStats);
      m_rank_minimum.resize(nStats);
      m_rank_maximum.resize(nStats);
      m_rank_std_dev.resize(nStats);

      for (size_t i = 0; i < nStats; ++i) {
        double val;

        if(InfoMapper<E, T>::m_counts[i] )
          val = InfoMapper<E, T>::m_values[i] / InfoMapper<E, T>::m_counts[i];
        else
          val = InfoMapper<E, T>::m_values[i];

        m_node_sum[i] = val;
        m_node_average[i] = val;

        m_rank_average[i] = val;
        m_rank_minimum[i] = double_int(val, 0);
        m_rank_maximum[i] = double_int(val, 0);
        m_rank_std_dev[i] = 0;
      }
    }
  };

  //______________________________________________________________________
  void reportSummmaryStats( const char* statsName,
                            const int timeStep,
                            const double simTime,
                            bool calcImbalance )
  {
    unsigned int nStats = InfoMapper<E, T>::m_keys.size();

    if( nStats )
    {
      std::ostringstream header;
      header << statsName << " performance summary stats for "
             << "time step " << timeStep
             << " at time="  << simTime
             << std::endl
             << "  " << std::left
             << std::setw(24) << "Description"
             << std::setw(18) << "Units";
      if (InfoMapper<E, T>::calculateMinimum()) {
        header << std::setw(18) << "Minimum"
               << std::setw(12) << "Rank";
      }
      if (InfoMapper<E, T>::calculateAverage()) {
        header << std::setw(18) << "Average";
      }
      if (InfoMapper<E, T>::calculateStdDev()) {
        header << std::setw(18) << "Std. Dev.";
      }
      if (InfoMapper<E, T>::calculateMaximum()) {
        header << std::setw(18) << "Maximum"
               << std::setw(12) << "Rank";
      }
      if( calcImbalance ) {
        header << std::setw(12) << "100*(1-ave/max) '% load imbalance'";
      }
    
      header << std::endl;
    
      std::ostringstream message;

      for (unsigned int i=0; i<InfoMapper<E, T>::size(); ++i) {
        if( InfoMapper<E, T>::getRankMaximum(i) != 0.0 )
        {
          if (message.str().size()) {
            message << std::endl;
          }
          
          message << "  " << std::left << std::setw(24) << InfoMapper<E, T>::getName(i) << "[" << std::setw(15)
                  << InfoMapper<E, T>::getUnits(i) << "]";
          
          if (InfoMapper<E, T>::calculateMinimum()) {
            message << " : " << std::setw(15) << InfoMapper<E, T>::getRankMinimum(i) << " : " << std::setw(9)
                    << InfoMapper<E, T>::getRankForMinimum(i);
          }
          
          if (InfoMapper<E, T>::calculateAverage()) {
            message << " : " << std::setw(15) << InfoMapper<E, T>::getRankAverage(i);
          }
          
          if (InfoMapper<E, T>::calculateStdDev()) {
            message << " : " << std::setw(15) << InfoMapper<E, T>::getRankStdDev(i);
          }
          
          if (InfoMapper<E, T>::calculateMaximum()) {
            message << " : " << std::setw(15) << InfoMapper<E, T>::getRankMaximum(i) << " : " << std::setw(9)
                    << InfoMapper<E, T>::getRankForMaximum(i);
          }
          
          if( calcImbalance ) {
            if( InfoMapper<E, T>::getRankMaximum(i) == 0.0 )
              message << "0";
            else
              message << 100.0 * (1.0 - (InfoMapper<E, T>::getRankAverage(i) /
                                         InfoMapper<E, T>::getRankMaximum(i)));
          }         
        }
      }
      
      if( message.str().size() ) {
        DOUT(true, header.str()+message.str());
      }
    }
  }
  
  //______________________________________________________________________
  void reportIndividualStats( const char* statsName,
                              const int rank,
                              const int timeStep,
                              const double simTime )
  {
    std::ostringstream header;
    header << "--" << std::left
           << "Rank: " << std::setw(5) << rank << "  "
           << statsName << " performance stats for "
           << "time step " << timeStep
           << " at time="  << simTime
           << std::endl;
    
    std::ostringstream message;

    unsigned int nStats = InfoMapper<E, T>::m_keys.size();
    
    for (unsigned int i=0; i<nStats; ++i) {
      if (InfoMapper<E, T>::getRankValue(i) > 0) {
        if (message.str().size()) {
          message << std::endl;
        }
        message << "  " << std::left
                << "Rank: " << std::setw(6) << rank
                << std::left << std::setw(24) << InfoMapper<E, T>::getName(i)
                << "["   << std::setw(15) << InfoMapper<E, T>::getUnits(i) << "]"
                << " : " << std::setw(15) << InfoMapper<E, T>::getRankValue(i);
        if( InfoMapper<E, T>::getCount(i) )
          message << " ("  << std::setw( 4) << InfoMapper<E, T>::getCount(i) << ")";
      }
    }
    
    if (message.str().size()) {
      DOUT(true, header.str() + message.str());
    }
  }
  
protected:
  bool calculate_average{true};
  bool calculate_minimum{false};
  bool calculate_maximum{true};
  bool calculate_std_dev{false};
  
  std::vector< double > m_node_sum;     // Sum of all ranks on a single node
  std::vector< double > m_node_average; // Average of all ranks on a single node
  
  std::vector< double >     m_rank_average;      // Average over all ranks
  std::vector< double_int > m_rank_minimum;      // Minimum over all ranks
  std::vector< double_int > m_rank_maximum;      // Maximum over all ranks
  std::vector< double >     m_rank_std_dev;      // Standard deviation over all ranks
};

template<class E, class T> class VectorInfoMapper
{
public:
  VectorInfoMapper()
  {
    m_average.clear();
    m_minimum.clear();
    m_maximum.clear();
    m_std_dev.clear();
  };

  virtual ~VectorInfoMapper() {};

  virtual void calculateAverage( bool val ) { calculate_average = val; }
  virtual void calculateMinimum( bool val ) { calculate_minimum = val; }
  virtual void calculateMaximum( bool val ) { calculate_maximum = val; }
  virtual void calculateStdDev ( bool val ) { calculate_std_dev = val; }
  
  virtual bool calculateAverage() { return calculate_average; }
  virtual bool calculateMinimum() { return calculate_minimum; }
  virtual bool calculateMaximum() { return calculate_maximum; }
  virtual bool calculateStdDev () { return calculate_std_dev; }
  
        InfoMapper<E, T>& operator[](unsigned int index)       { return m_vecInfoMapper[index]; };
  const InfoMapper<E, T>& operator[](unsigned int index) const { return m_vecInfoMapper[index]; };

  virtual void clear()
  {
    m_vecInfoMapper.clear();

    m_average.clear();
    m_minimum.clear();
    m_maximum.clear();
    m_std_dev.clear();
  };

  virtual size_t size()
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
     
    m_average.push_back(0);
    m_minimum.push_back(double_int(0,-1));
    m_maximum.push_back(double_int(0,-1));
    m_std_dev.push_back(0);
  }

  // Get average over all entries
  virtual double getAverage( const E key )
  {
    m_vecInfoMapper[0].validKey( key );

    return m_average[ m_vecInfoMapper[0].m_keys[key] ];
  };

  virtual double getAverage( const unsigned int index )
  {
    const E key = m_vecInfoMapper[0].getKey(index);

    return getAverage( key );
  };

  virtual double getAverage( const std::string name )
  {
    const E key = m_vecInfoMapper[0].getKey(name);

    return getAverage( key );
  };

  // Get minium over all entries
  virtual double getMinimum( const E key )
  {
    m_vecInfoMapper[0].validKey( key );

    return m_minimum[ m_vecInfoMapper[0].m_keys[key] ].val;
  };

  virtual double getMinimum( const unsigned int index )
  {
    const E key = m_vecInfoMapper[0].getKey(index);

    return getMinimum( key );
  };

  virtual double getMinimum( const std::string name )
  {
    const E key = m_vecInfoMapper[0].getKey(name);

    return getMinimum( key );
  };

  // Get index for the minimum entry
  virtual unsigned int getIndexForMinimum( const E key )
  {
    m_vecInfoMapper[0].validKey( key );

    return m_minimum[ m_vecInfoMapper[0].m_keys[key] ].rank;
  };

  virtual unsigned int getIndexForMinimum( const unsigned int index )
  {
    const E key = m_vecInfoMapper[0].getKey(index);

    return getIndexForMinimum( key );
  };

  virtual unsigned int getIndexForMinimum( const std::string name )
  {
    const E key = m_vecInfoMapper[0].getKey(name);

    return getIndexForMinimum( key );
  };

  // Get maximum over all entries
  virtual double getMaximum( const E key )
  {
    m_vecInfoMapper[0].validKey( key );

    return m_maximum[ m_vecInfoMapper[0].m_keys[key] ].val;
  };

  virtual double getMaximum( const unsigned int index )
  {
    const E key = m_vecInfoMapper[0].getKey(index);

    return getMaximum( key );
  };

  virtual double getMaximum( const std::string name )
  {
    const E key = m_vecInfoMapper[0].getKey(name);

    return getMaximum( key );
  };

  // Get index for the maximum entry
  virtual unsigned int getIndexForMaximum( const E key )
  {
    m_vecInfoMapper[0].validKey( key );

    return m_maximum[ m_vecInfoMapper[0].m_keys[key] ].rank;
  };

  virtual unsigned int getIndexForMaximum( const unsigned int index )
  {
    const E key = m_vecInfoMapper[0].getKey(index);

    return getIndexForMaximum( key );
  };

  virtual unsigned int getIndexForMaximum( const std::string name )
  {
    const E key = m_vecInfoMapper[0].getKey(name);

    return getIndexForMaximum( key );
  };

  // Get std dev over all entries
  virtual double getStdDev( const E key )
  {
    m_vecInfoMapper[0].validKey( key );

    return m_std_dev[ m_vecInfoMapper[0].m_keys[key] ];
  };

  virtual double getStdDev( const unsigned int index )
  {
    const E key = m_vecInfoMapper[0].getKey(index);

    return getStdDev( key );
  };

  virtual double getStdDev( const std::string name )
  {
    const E key = m_vecInfoMapper[0].getKey(name);

    return getStdDev( key );
  };

  // Reduce
  virtual void reduce( bool skipFirst )
  {
    unsigned int nStats = m_vecInfoMapper[0].m_keys.size();

    if (nStats == 0) {
      return;
    }

    m_average.resize(nStats);
    m_minimum.resize(nStats);
    m_maximum.resize(nStats);
    m_std_dev.resize(nStats);

    if (m_vecInfoMapper.size() > 1) {

      // Calculate each stat.
      for (size_t i = 0; i < nStats; ++i) {

        m_minimum[i] = double_int( DBL_MAX, -1);
        m_maximum[i] = double_int(-DBL_MAX, -1);

        // Calculate across all entries.
        for (size_t j = int(skipFirst); j < m_vecInfoMapper.size(); ++j) {
        
          double val;
          if( m_vecInfoMapper[j].m_counts[i] )
            val = m_vecInfoMapper[j].m_values[i] / m_vecInfoMapper[j].m_counts[i];
          else
            val = m_vecInfoMapper[j].m_values[i];

          // Sum across all entires.
          if( calculate_average || calculate_std_dev )
            m_average[i] += val;

          // Min across all entries.
          if( calculate_minimum ) {
            if( val < m_minimum[i].val )
               m_minimum[i] = double_int(val, j);
          }

          // Max across all entries.
          if( calculate_maximum ) {
            if( m_maximum[i].val < val )
               m_maximum[i] = double_int(val, j);
          }
        }

        // Sums across all entries.
        if( calculate_average || calculate_std_dev )
        {
          // Calculate the average.
          m_average[i] /= (m_vecInfoMapper.size() - int(skipFirst));
          
          if( calculate_std_dev )
          {
            //  Calculate the sum of squared differences
            double sum = 0;
            for (size_t j = int(skipFirst); j < m_vecInfoMapper.size(); ++j) {
        
              double val;
              if( m_vecInfoMapper[j].m_counts[i] )
                val = m_vecInfoMapper[j].m_values[i] / m_vecInfoMapper[j].m_counts[i];
              else
                val = m_vecInfoMapper[j].m_values[i];
              
              sum = (val - m_average[i]) * (val - m_average[i]);
            }

            m_std_dev[i] = std::sqrt(sum / (m_vecInfoMapper.size() - int(skipFirst) - 1) );  
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

        m_average[i] = val;
        m_minimum[i] = double_int(val, 0);
        m_maximum[i] = double_int(val, 0);
        m_std_dev[i] = 0;
      }
    }
  };

  //______________________________________________________________________
  //
  void reportSummaryStats( const char* statsName,
                           const int rank,
                           const int timeStep,
                           const double simTime,
                           bool calcImbalance )
  {
    unsigned int nStats = m_vecInfoMapper[0].size();

    if( nStats )
    {
      std::ostringstream header;
      header << "Rank: " << std::setw(6) << rank << "  "
             << statsName << " performance summary stats for "
             << "time step " << timeStep
             << " at time="  << simTime
             << std::endl
             << "  " << std::left
             << std::setw(24) << "Description"
             << std::setw(18) << "Units";
      if (calculateMinimum()) {
        header << std::setw(18) << "Minimum"
               << std::setw(12) << "Thread";
      }
      if (calculateAverage()) {
        header << std::setw(18) << "Average";
      }
      if (calculateStdDev()) {
        header << std::setw(18) << "Std. Dev.";
      }
      if (calculateMaximum()) {
        header << std::setw(18) << "Maximum"
               << std::setw(12) << "Thread";
      }
      if( calcImbalance ) {
        header << std::setw(12) << "100*(1-ave/max) '% load imbalance'";
      }
      
      header << std::endl;
      
      std::ostringstream message;
      
      for (unsigned int i=0; i<nStats; ++i) {
        if( getMaximum(i) != 0.0 )
        {
          if (message.str().size()) {
            message << std::endl;
          }
          
          message << "  " << std::left << std::setw(24) << m_vecInfoMapper[0].getName(i) << "[" << std::setw(15)
                  << m_vecInfoMapper[0].getUnits(i) << "]";
          
          if (calculateMinimum()) {
            message << " : " << std::setw(15) << getMinimum(i) << " : " << std::setw(9)
                    << getIndexForMinimum(i);
          }
          
          if (calculateAverage()) {
            message << " : " << std::setw(15) << getAverage(i);
          }
          
          if (calculateStdDev()) {
            message << " : " << std::setw(15) << getStdDev(i);
          }
          
          if (calculateMaximum()) {
            message << " : " << std::setw(15) << getMaximum(i) << " : " << std::setw(9)
                    << getIndexForMaximum(i);
          }
          
          if( calcImbalance ) {
            if( getMaximum(i) == 0.0 )
              message << "0";
            else
              message << 100.0 * (1.0 - (getAverage(i) / getMaximum(i)));
          }           
        }
      }
      
      if( message.str().size() ) {
        DOUT(true, header.str()+message.str());
      }
    }
  }

  //______________________________________________________________________
  //
  void reportIndividualStats( const char* statsName,
                              const int rank,
                              const int timeStep,
                              const double simTime )
  {
    std::ostringstream header;
    header << "--" << std::left
           << "Rank: " << std::setw(4) << rank << "  "
           << statsName << " performance stats for "
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
                  << "Rank: " << std::setw(6) << rank
                  << "Thread: " << std::setw(6) << j
                  << std::left << std::setw(24) << m_vecInfoMapper[j].getName(i)
                  << "["   << std::setw(15) << m_vecInfoMapper[j].getUnits(i) << "]"
                  << " : " << std::setw(15) << m_vecInfoMapper[j].getRankValue(i);
          if( m_vecInfoMapper[j].getCount(i) )
            message << " ("  << std::setw( 4) << m_vecInfoMapper[j].getCount(i) << ")";
        }
      }
    }
  
    if (message.str().size()) {
      DOUT(true, header.str() + message.str());
    }
  }

protected:
  std::vector< InfoMapper<E, T> > m_vecInfoMapper;

  bool calculate_average{true};
  bool calculate_minimum{false};
  bool calculate_maximum{true};
  bool calculate_std_dev{false};
  
  std::vector< double >     m_average;      // Average over all entries
  std::vector< double_int > m_minimum;      // Minimum over all entries
  std::vector< double_int > m_maximum;      // Maximum over all 
  std::vector< double >     m_std_dev;      // Standard deviation over all entries
};

} // End namespace Uintah

#endif // CORE_UTIL_INFOMAPPER_H
