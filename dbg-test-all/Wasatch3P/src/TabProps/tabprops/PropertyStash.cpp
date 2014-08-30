/*
 * The MIT License
 *
 * Copyright (c) 2010-2014 The University of Utah
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
 */#include "PropertyStash.h"
#include <tabprops/Archive.h>

#include <sstream>
#include <stdexcept>


//-- Boost serialization tools --//
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/map.hpp>

//-------------------------------------------------------------------

template<typename T>
void
PropertyStash::set( const std::string key, const T t )
{
  std::pair<typename Stash::iterator,bool> result = stash_.insert(make_pair(key,t));
  if( !(result.second) ){
    std::ostringstream msg;
    msg << "ERROR: could not set property '" << key << "' because an entry already exists" << std::endl
        << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error(msg.str());
  }
}

//-------------------------------------------------------------------

template<typename T>
T
PropertyStash::get( const std::string& key ) const
{
  const Stash::const_iterator i = stash_.find(key);
  if( i == stash_.end() ){
      std::ostringstream msg;
      msg << "property '" << key << "' not found!" << std::endl
          << "         " << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::invalid_argument( msg.str() );
  }
  return boost::get<T>( i->second );
}

//-------------------------------------------------------------------

template<typename Archive>
void
PropertyStash::serialize( Archive& ar, const unsigned int version )
{
  ar & BOOST_SERIALIZATION_NVP( stash_ );
}

//-------------------------------------------------------------------


//===================================================================
// Explicit instantiations

#define INSTANTIATE( T )                                                \
  template T    PropertyStash::get<T>(const std::string&) const;        \
  template void PropertyStash::set<T>(const std::string, const T);

INSTANTIATE( double );
INSTANTIATE( int );
INSTANTIATE( std::string );

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

template void PropertyStash::serialize<OutputArchive>( OutputArchive&, const unsigned int );
template void PropertyStash::serialize<InputArchive >( InputArchive&,  const unsigned int );

//===================================================================
