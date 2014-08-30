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

#ifndef TabProps_Archive_h
#define TabProps_Archive_h

#include <tabprops/TabPropsConfig.h>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>  // for multi_array
#include <boost/multi_array.hpp>

/*
 * Define the type of archive we want.
 */
//# define ASCII_IO     // define this to use ASCII archives (definitely portable)
#define BINARY_IO  // define this to use binary archives (maybe portable?)
//#define XML_IO     // define this to use XML archives (bloated)

# ifdef BINARY_IO

#  include <boost/archive/binary_oarchive.hpp>
#  include <boost/archive/binary_iarchive.hpp>
   typedef boost::archive::binary_oarchive OutputArchive;
   typedef boost::archive::binary_iarchive InputArchive;

# elif defined ASCII_IO

#  include <boost/archive/text_oarchive.hpp>
#  include <boost/archive/text_iarchive.hpp>
   typedef boost::archive::text_oarchive OutputArchive;
   typedef boost::archive::text_iarchive InputArchive;

# elif defined XML_IO

#  include <boost/archive/xml_oarchive.hpp>
#  include <boost/archive/xml_iarchive.hpp>
   typedef boost::archive::xml_oarchive OutputArchive;
   typedef boost::archive::xml_iarchive InputArchive;

# else

#  error "NO VALID IO SCHEME DEFINED"

# endif

//===================================================================
namespace boost {
namespace serialization {

  // Serialization for boost multi_array objects:
  template<typename Archive,typename T,size_t Dim>
  void save( Archive& ar, const boost::multi_array<T,Dim>& t, const unsigned int version )
  {
    ar << boost::serialization::make_array( t.shape(), Dim )
       << boost::serialization::make_array( t.data(), t.num_elements() );
  }

  template<typename T,typename Archive,size_t Dim>
  void load( Archive& ar, boost::multi_array<T,Dim>& t, const unsigned int version )
  {
    boost::array<typename boost::multi_array<T,Dim>::index,Dim> shape;
    ar >> boost::serialization::make_array( shape.data(), Dim );

    t.resize(shape);
    ar >> boost::serialization::make_array( t.data(), t.num_elements() );
  }

  template<typename Archive,typename T,size_t Dim>
  void serialize( Archive& ar, boost::multi_array<T,Dim>& t, const unsigned int version )
  {
    boost::serialization::split_free( ar, t, version );
  }

  // serialization for std::pair<T,T> objects
  template<typename Archive,typename T>
  void save( Archive& ar, const std::pair<T,T>& p, const unsigned int version )
  {
    ar << p.first << p.second;
  }
  template<typename Archive,typename T>
  void load( Archive& ar, std::pair<T,T>& p, const unsigned int version )
  {
    ar >> p.first >> p.second;
  }
  template<typename Archive,typename T>
  void serialize( Archive& ar, std::pair<T,T>& p, const unsigned int version )
  {
    boost::serialization::split_free( ar, p, version );
  }

} // namespace serialization
} // namespace boost
//===================================================================

#endif // TabProps_Archive_h
