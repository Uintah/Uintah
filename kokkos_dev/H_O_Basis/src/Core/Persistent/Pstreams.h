/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  Pstream.h: reading/writing persistent objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Pstream_h
#define SCI_project_Pstream_h 1

#include <Core/Persistent/Persistent.h>
#include <stdio.h>

#include <zlib.h>

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

class BinaryPiostream : public Piostream {
protected:
  FILE* fp_;

  virtual const char *endianness();

private:
  template <class T> void gen_io(T&, const char *);

public:
  BinaryPiostream(const string& filename, Direction dir,
                  const int& v = -1, ProgressReporter *pr = 0);
  BinaryPiostream(int fd, Direction dir, const int& v = -1,
                  ProgressReporter *pr = 0);
  virtual ~BinaryPiostream();

  virtual void io(char&);
  virtual void io(signed char&);
  virtual void io(unsigned char&);
  virtual void io(short&);
  virtual void io(unsigned short&);
  virtual void io(int&);
  virtual void io(unsigned int&);
  virtual void io(long&);
  virtual void io(unsigned long&);
  virtual void io(long long&);
  virtual void io(unsigned long long&);
  virtual void io(double&);
  virtual void io(float&);
  virtual void io(string& str);

  virtual bool supports_block_io() { return (version() > 1); }
  virtual bool block_io(void*, size_t, size_t);
};


class BinarySwapPiostream : public BinaryPiostream {
protected:
  virtual const char *endianness();

private:
  template <class T> void gen_io(T&, const char *);

public:
  BinarySwapPiostream(const string& filename, Direction d,
                      const int& v = -1, ProgressReporter *pr = 0);
  BinarySwapPiostream(int fd, Direction dir, const int& v = -1,
                      ProgressReporter *pr = 0);
  virtual ~BinarySwapPiostream();

  virtual void io(short&);
  virtual void io(unsigned short&);
  virtual void io(int&);
  virtual void io(unsigned int&);
  virtual void io(long&);
  virtual void io(unsigned long&);
  virtual void io(long long&);
  virtual void io(unsigned long long&);
  virtual void io(double&);
  virtual void io(float&);

  virtual bool supports_block_io() { return false; }
  virtual bool block_io(void*, size_t, size_t) { return false; }
};


class TextPiostream : public Piostream {
private:
  std::istream* istr;
  std::ostream* ostr;
  bool ownstreams_p_;

  void expect(char);
  virtual void emit_pointer(int&, int&);
  void io(int, string& str);

public:
  TextPiostream(const string& filename, Direction dir,
                ProgressReporter *pr = 0);
  TextPiostream(std::istream *strm, ProgressReporter *pr = 0);
  TextPiostream(std::ostream *strm, ProgressReporter *pr = 0);
  virtual ~TextPiostream();

  virtual string peek_class();
  virtual int begin_class(const string& name, int);
  virtual void end_class();
  virtual void begin_cheap_delim();
  virtual void end_cheap_delim();

  virtual void io(bool&);
  virtual void io(char&);
  virtual void io(signed char&);
  virtual void io(unsigned char&);
  virtual void io(short&);
  virtual void io(unsigned short&);
  virtual void io(int&);
  virtual void io(unsigned int&);
  virtual void io(long&);
  virtual void io(unsigned long&);
  virtual void io(long long&);
  virtual void io(unsigned long long&);
  virtual void io(double&);
  virtual void io(float&);
  virtual void io(string& str);
};


//! The Fast stream is binary, its results can only safely be used
//! on the architecture where the file is generated.
class FastPiostream : public Piostream {
private:
  FILE* fp_;

  void report_error(const char *);
  template <class T> void gen_io(T&, const char *);

public:
  FastPiostream(const string& filename, Direction dir,
                ProgressReporter *pr = 0);
  FastPiostream(int fd, Direction dir,
                ProgressReporter *pr = 0);
  virtual ~FastPiostream();

  virtual void io(bool&);
  virtual void io(char&);
  virtual void io(signed char&);
  virtual void io(unsigned char&);
  virtual void io(short&);
  virtual void io(unsigned short&);
  virtual void io(int&);
  virtual void io(unsigned int&);
  virtual void io(long&);
  virtual void io(unsigned long&);
  virtual void io(long long&);
  virtual void io(unsigned long long&);
  virtual void io(double&);
  virtual void io(float&);
  virtual void io(string& str);

  virtual bool supports_block_io() { return true; }
  virtual bool block_io(void*, size_t, size_t);
};


} // End namespace SCIRun


#endif
