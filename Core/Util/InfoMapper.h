/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef UINTAH_HOMEBREW_Stats_H
#define UINTAH_HOMEBREW_Stats_H

#include <Core/Exceptions/InternalError.h>

#include <iostream>
#include <sstream>
#include <map>


namespace Uintah {

template<class E, class T> class InfoMapper
{
public:
  InfoMapper()
  {
    d_values.clear();
    d_names.clear();
  };

  ~InfoMapper() {};
    
  size_t size()
  {
    return d_values.size();
  };

  E maxStats()
  {
    return (E) d_values.size();
  };

  void clear()
  {
    d_values.clear();
    d_names.clear();
  };
  
  void reset( T val )
  {
    for( int i=0; i<d_values.size(); ++i )
      d_values[(E) i] = val;
  };

  void validStat( E key )
  {
    if( !exists( key ) )
    {
      std::stringstream msg;
      msg << "Requesting an undefined stat key (" << key << ") ";
      
      throw SCIRun::InternalError( msg.str(), __FUNCTION__, __LINE__);
    }
  };

  bool exists( E key )
  {
    return (d_values.find( key ) != d_values.end());
  };

  void validate( E lastKey )
  {
    if( d_values.size() != (int) lastKey )
    {
      std::stringstream msg;
      msg << "The stat count does not match. Expected "
	  << (int) lastKey << " stats. But added "
	  << (int) d_values.size() << " stats.";
      
      throw SCIRun::InternalError(msg.str(), __FILE__, __LINE__);
    }
  };
  
  void addStat( E key, std::string name )
  {
    if( !exisits( key ) )
    {
      d_values[key];
      d_names[key] = name;
    }
    else
    {
      std::stringstream msg;
      msg << "Adding a key (" << key << ") with name, " 
	  << name << " that already exists.";

      throw SCIRun::InternalError( msg.str(), __FUNCTION__, __LINE__);
    }
  };
  
  void addStat( E key, std::string name, T value )
  {
    if( !exists( key ) )
    {
      d_values[key] = value;
      d_names[key] = name;
    }
    else
    {
      std::stringstream msg;
      msg << "Adding a key (" << key << ") with name, " 
	  << name << " that already exists.";

      throw SCIRun::InternalError( msg.str(), __FUNCTION__, __LINE__);
    }
  };
  
  void deleteStat( E key )
  {
    typename std::map< E, T >::iterator           eIter = d_values.find( key );
    typename std::map< E, std::string >::iterator sIter = d_names.find( key );
    
    if( eIter != d_values.end() && sIter != d_names.end() )
    {
      d_values.erase(key);
      d_names.erase(key);
    }
    else
    {
      std::stringstream msg;
      msg << "Trying to delete a key (" << key << ") that does not exist.";
      throw SCIRun::InternalError( msg.str(), __FUNCTION__, __LINE__);
    }
  }

        T& operator[](E idx)       { return d_values[idx]; };
  const T& operator[](E idx) const { return d_values[idx]; };
  
  void setStat( E key, T value )
  {
    d_values[key] = value;
  };

  T getStat( E key )
  {
    validStat( key );

    return d_values[key];
  };
  
  T getStat( std::string name )
  {
    E key = getEnum(name);
    
    validStat( key );

    return d_values[ key ];
  };
  
  std::string getName( E key )
  {
    validStat( key );

    return d_names[key];
  };

  E getEnum( std::string name )
  {
    for( int i=0; i<d_values.size(); ++i )
    {
      if( name.find( d_names[(E) i] ) != std::string::npos )
	return (E) i;
    }

    return (E) d_values.size();
  };
  
private:  
  std::map< E, T > d_values;
  std::map< E, std::string > d_names;
};

} // End namespace Uintah

#endif
