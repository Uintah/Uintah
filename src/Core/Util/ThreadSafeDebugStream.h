/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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


#ifndef CORE_UTIL_THREADSAFEDEBUGSTREAM_H
#define CORE_UTIL_THREADSAFEDEBUGSTREAM_H

#include <cstdlib> // for getenv()
#include <cstdio>  // for printf()
#include <string>
#include <sstream>
#include <ostream>
#include <iostream>

namespace Uintah {

class ThreadSafeDebugStream {

  private:

    // identifies me uniquely
    std::string m_name;

    std::ostringstream m_out;

    // my default action (used if nothing is specified in SCI_DEBUG)
    bool m_default_on;

    // check the environment variable
    bool checkenv();


  public:

    ThreadSafeDebugStream();

    ThreadSafeDebugStream(const std::string& name, bool defaulton = true);

    ~ThreadSafeDebugStream();

    ThreadSafeDebugStream& operator <<(std::iostream& arg)
    {
      if (m_default_on || checkenv()) {
        m_out << arg;
      }
      return *this;
    }

    void flush() {
      printf("%s", m_out.str().c_str());
      m_out.clear();
    }

    struct replace
    {
      void operator()(char& c)
      {
        if (c == ':') {
          c = ' ';
        }
      }
    };

};

}  // End namespace Uintah

#endif // end CORE_UTIL_THREADSAFEDEBUGSTREAM_H
