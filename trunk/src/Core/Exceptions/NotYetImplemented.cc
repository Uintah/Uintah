/*
 * NotYetImplemented.cc
 *
 *  Created on: Jun 6, 2018
 *      Author: jbhooper
 */

#include <Core/Exceptions/NotYetImplemented.h>
#include <sstream>
#include <iostream>

namespace Uintah {
NotYetImplemented::NotYetImplemented(const std::string  & referencingObject
                                    ,const std::string  & referencedMethod
                                    ,const std::string  & extraInfo
                                    ,const char         * file
                                    ,      int            line)

  {
    std::ostringstream sOut;

    sOut << "A requested capability has not yet been implemented in code.\n"
         << "  Calling location: " << file << ":" << line << "\n"
         << "  Calling object: " << referencingObject << "\n"
         << "  Called method: " << referencedMethod << "\n";

    if (extraInfo != "")
    {
      sOut << " -- Additional contextual information: --\n ("
           << extraInfo << ")\n";
    }

    m_msg = sOut.str();
#ifdef EXCEPTIONS_CRASH
    std::cout << m_msg << "\n";
#endif

  }
NotYetImplemented::NotYetImplemented( const NotYetImplemented & copy)
                                    : m_msg(copy.m_msg)
{

}

NotYetImplemented::~NotYetImplemented()
{

}

const char* NotYetImplemented::message() const
{
  return m_msg.c_str();
}

const char* NotYetImplemented::type() const
{
  return "Uintah::Exceptions::NotYetImplemented";
}

}
