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

} // End namespace Uintah

#endif // CORE_UTIL_INFOMAPPER_H
