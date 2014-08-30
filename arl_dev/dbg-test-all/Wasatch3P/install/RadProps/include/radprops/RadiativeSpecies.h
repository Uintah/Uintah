/**
 *  \file   RadiativeSpecies.h
 *
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
#ifndef RadiativeSpecies_h
#define RadiativeSpecies_h

#include <string>
#include <sstream>
#include <stdexcept>


/**
 *  \enum RadiativeSpecies
 *  \brief supported species
 */
enum RadiativeSpecies { H2O, CO2, CO, NO, OH };

/**
 * @brief given the RadiativeSpecies enum, obtain the corresponding text string name
 * @param sp the enumerated value
 * @return the string name
 */
inline std::string species_name( const RadiativeSpecies sp )
{
  std::string name;
  switch (sp) {
  case H2O: name="H2O"; break;
  case CO2: name="CO2"; break;
  case CO : name="CO" ; break;
  case NO : name="NO" ; break;
  case OH : name="OH" ; break;
  default: name="" ;
  std::ostringstream msg;
  msg << __FILE__ << " : " << __LINE__ << std::endl
      << "ERROR! Cannot translate the requested enum to a name." << std::endl
      << "       It is not in the list of radiative species." << std::endl;
  throw std::runtime_error( msg.str() );

  break;
  }
  return name;
}

/**
 * @brief given the name, obtain the corresponding enum (if a match is available)
 * @param name the string name for the species
 * @return the enumerated value
 */
inline RadiativeSpecies species_enum( const std::string& name )
{
  RadiativeSpecies rs;
  if( name.compare("H2O")== 0 ) {rs=H2O;}
  else if( name.compare("CO2")== 0 ) {rs=CO2;}
  else if( name.compare("CO" )== 0 ) {rs=CO;}
  else if( name.compare("NO" )== 0 ) {rs=NO;}
  else if( name.compare("OH" )== 0 ) {rs=OH;}
  else {
    std::ostringstream msg;
    msg << "ERROR! '" << name << "' is not in the list of supported radiative species." << std::endl
        << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( msg.str() );
  }
  return rs;
}


#endif // RadiativeSpecies_h
