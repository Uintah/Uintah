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

#ifndef CORE_UTIL_DOUT_HPP
#define CORE_UTIL_DOUT_HPP

#include <Core/Exceptions/InternalError.h>
#include <Core/Parallel/UintahMPI.h>

#include <sci_defs/compile_defs.h> // for STATIC_BUILD

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>


//__________________________________
// Conditional
#define DOUT( cond, ... )                                  \
  if (cond) {                                              \
    std::ostringstream dout_msg;                           \
    dout_msg << __VA_ARGS__;                               \
    printf("%s\n",dout_msg.str().c_str());                 \
  }

//__________________________________
// Conditional with rank
#define DOUTR( cond, ... )                                 \
  if ( cond) {                                             \
    std::ostringstream dout_msg;                           \
    dout_msg << __VA_ARGS__;                               \
    dout_msg << "Rank ";                                   \
    dout_msg << MPI::Impl::prank( MPI_COMM_WORLD ) << " "; \
    printf("%s\n",dout_msg.str().c_str());                 \
  }

//__________________________________
// Conditional with rank, file, and line number reported
#define DOUTALL( cond, ... )                               \
  if (cond) {                                              \
    std::ostringstream pout_msg;                           \
    pout_msg << "Rank ";                                   \
    pout_msg << MPI::Impl::prank( MPI_COMM_WORLD ) << " "; \
    pout_msg << __FILE__ << ":";                           \
    pout_msg << __LINE__ << " : ";                         \
    pout_msg << __VA_ARGS__;                               \
    printf("%s\n", pout_msg.str().c_str());                \
  }

//__________________________________
//  For threads, Rank and thread reported
#define TOUT( ... ) {                                       \
    std::ostringstream tout_msg;                            \
    tout_msg << "Rank ";                                    \
    tout_msg << MPI::Impl::prank( MPI_COMM_WORLD ) << "  "; \
    tout_msg << "Thread ";                                  \
    tout_msg << MPI::Impl::tid() << ") ";                   \
    tout_msg << __VA_ARGS__;                                \
    printf("%s\n", tout_msg.str().c_str());                 \
  }

//__________________________________
//  For threads, with rank, thread, file, and line, reported
#define TOUTALL( ... ) {                                    \
    std::ostringstream tout_msg;                            \
    tout_msg << "Rank ";                                    \
    tout_msg << MPI::Impl::prank( MPI_COMM_WORLD ) << "  "; \
    tout_msg << "Thread ";                                  \
    tout_msg << MPI::Impl::tid() << ") ";                   \
    tout_msg << __FILE__ << ":";                            \
    tout_msg << __LINE__ << " : ";                          \
    tout_msg << __VA_ARGS__;                                \
    printf("%s\n", tout_msg.str().c_str());                 \
  }


namespace Uintah {

class Dout
{

public:

  Dout()               = default;
  Dout( const Dout & ) = default;
  Dout( Dout && )      = default;

  Dout & operator=( const Dout & ) = default;
  Dout & operator=( Dout && )      = default;

  Dout( std::string const & name
      , std::string const & component
      , std::string const & description
      , bool default_active
      )
    : m_active{ is_active(name, default_active) }
    , m_name{ name }
    , m_component{ component }
    , m_description{ description }
  {

#ifdef STATIC_BUILD
    if (!m_all_douts_initialized) {
      m_all_douts_initialized = true;
      m_all_douts = std::map<std::string, Dout*>();
    }
#endif

    auto iter = m_all_douts.find(m_name);
    if (iter != m_all_douts.end()) {

      printf("These two dout are for the same component and have the same name. \n");
      (*iter).second->print();
      print();
      
      // Two Douts for the same component with the same name.
      SCI_THROW(InternalError(std::string("Multiple Douts for component " + m_component + " with name " + m_name), __FILE__, __LINE__));
    }
    m_all_douts[m_name] = this;
  }

  explicit operator bool() const { return m_active; }

  const std::string & name ()        const { return m_name; }
  const std::string & component ()   const { return m_component; }
  const std::string & description () const { return m_description; }

  friend bool operator<(const Dout & a, const Dout &b ) { return a.m_name < b.m_name; }

  // Note these two methods were added at the request of developers.
  //   active() is effectively the same as operator bool() - APH, 07/14/17
  bool active() const { return m_active; }
  void setActive( bool active ) { m_active = active; }

  void print() const
  {
    std::stringstream message;
    message << std::setw(2)  << std::left << (m_active ? "+" : "-")
            << std::setw(30) << std::left << m_name.c_str()
            << std::setw(75) << std::left << m_description.c_str()
            << std::setw(30) << std::left << m_component.c_str()
            << std::endl;
    printf("%s", message.str().c_str());
  }

  static void printAll()
  {
    printf("--------------------------------------------------------------------------------\n");
    for (auto iter = m_all_douts.begin(); iter != m_all_douts.end(); ++iter) {
      (*iter).second->print();
    }
    printf("--------------------------------------------------------------------------------\n\n");
  }

  static std::map<std::string, Dout*> m_all_douts;
  static bool                         m_all_douts_initialized;


private:

  static bool is_active( std::string const& arg_name, bool default_active )
  {
    const char * sci_debug = std::getenv("SCI_DEBUG");
    const std::string tmp = "," + arg_name + ":";
    const char * name = tmp.c_str();
    size_t n = tmp.size();

    if ( !sci_debug ) {
      return default_active;
    }

    const char * sub =  strstr(sci_debug, name);
    if ( !sub ) {
      sub = strstr( sci_debug, name+1);
      --n;
      // name not found
      if ( !sub || sub != sci_debug ) {
        return default_active;
      }
    }
    return sub[n] == '+';
  }


private:

  bool         m_active      {false};
  std::string  m_name        {""};
  std::string  m_component   {""};
  std::string  m_description {""};
};

} // namespace Uintah

#endif //UINTAH_CORE_UTIL_DOUT_HPP
