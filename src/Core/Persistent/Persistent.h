/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

/*
 *  Persistent.h: Base class for persistent objects...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 */

#ifndef SCI_project_Persistent_h
#define SCI_project_Persistent_h 1

#include <map>
#include <string>

#include <Core/Util/Assert.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Persistent/share.h>
namespace SCIRun {

using std::string;
using std::map;
using std::pair;

class Persistent;

//----------------------------------------------------------------------
struct SCISHARE PersistentTypeID {
  string type;
  string parent;
  Persistent* (*maker)();
  PersistentTypeID(const string& type, 
		   const string& parent,
		   Persistent* (*maker)(), 
		   Persistent* (*bc_maker1)() = 0, 
		   Persistent* (*bc_maker2)() = 0);
  ~PersistentTypeID();
  Persistent* (*bc_maker1)();
  Persistent* (*bc_maker2)();
};

//----------------------------------------------------------------------
class SCISHARE Piostream {
  
public:

  typedef map<Persistent*, int>			MapPersistentInt;
  typedef map<int, Persistent*>			MapIntPersistent;
  typedef map<string, PersistentTypeID*>	MapStringPersistentTypeID;

  enum Direction {
    Read,
    Write
  };

  enum Endian {
    Big,
    Little
  };

  static const int PERSISTENT_VERSION;
  void flag_error() { err = 1; }
  
protected:
  Piostream(Direction, int, const string &, ProgressReporter *pr);

  Direction dir;
  int version_;
  bool err;
  int file_endian;
  
  MapPersistentInt* outpointers;
  MapIntPersistent* inpointers;
  
  int current_pointer_id;

  bool have_peekname_;
  string peekname_;

  ProgressReporter *reporter_;
  bool own_reporter_;
  bool backwards_compat_id_;
  virtual void emit_pointer(int& have_data, int& pointer_id);
  static bool readHeader(ProgressReporter *pr,
                         const string& filename, char* hdr,
			 const char* type, int& version, int& endian);

  virtual void reset_post_header() = 0;
public:
  string file_name;

  virtual ~Piostream();

  virtual string peek_class();
  virtual int begin_class(const string& name, int current_version);
  virtual void end_class();
  virtual void begin_cheap_delim();
  virtual void end_cheap_delim();

  virtual void io(bool&);
  virtual void io(char&) = 0;
  virtual void io(signed char&) = 0;
  virtual void io(unsigned char&) = 0;
  virtual void io(short&) = 0;
  virtual void io(unsigned short&) = 0;
  virtual void io(int&) = 0;
  virtual void io(unsigned int&) = 0;
  virtual void io(long&) = 0;
  virtual void io(unsigned long&) = 0;
  virtual void io(long long&) = 0;
  virtual void io(unsigned long long&) = 0;
  virtual void io(double&) = 0;
  virtual void io(float&) = 0;
  virtual void io(string& str) = 0;

  void io(Persistent*&, const PersistentTypeID&);

  bool reading() const { return dir == Read; }
  bool writing() const { return dir == Write; }
  bool error() const { return err; }
  int version() const { return version_; }
  bool backwards_compat_id() const { return backwards_compat_id_; }
  void set_backwards_compat_id(bool p) { backwards_compat_id_ = p; }
  virtual bool supports_block_io() { return false; } // deprecated, redundant.
  // Returns true if bkock_io was supported (even on error).
  virtual bool block_io(void*, size_t, size_t) { return false; }

  SCISHARE friend Piostream* auto_istream(const string& filename,
                                 ProgressReporter *pr);
  SCISHARE friend Piostream* auto_ostream(const string& filename, const string& type,
                                 ProgressReporter *pr);
};

  SCISHARE Piostream* auto_istream(const string& filename,
                                   ProgressReporter *pr = 0);
  SCISHARE Piostream* auto_ostream(const string& filename, const string& type,
                                   ProgressReporter *pr = 0);


//----------------------------------------------------------------------
class SCISHARE Persistent {
public:
  virtual ~Persistent();
  virtual void io(Piostream&) = 0;
};

//----------------------------------------------------------------------
inline void Pio(Piostream& stream, bool& data) { stream.io(data); }
inline void Pio(Piostream& stream, char& data) { stream.io(data); }
inline void Pio(Piostream& stream, signed char& data) { stream.io(data); }
inline void Pio(Piostream& stream, unsigned char& data) { stream.io(data); }
inline void Pio(Piostream& stream, short& data) { stream.io(data); }
inline void Pio(Piostream& stream, unsigned short& data) { stream.io(data); }
inline void Pio(Piostream& stream, int& data) { stream.io(data); }
inline void Pio(Piostream& stream, unsigned int& data) { stream.io(data); }
inline void Pio(Piostream& stream, long& data) { stream.io(data); }
inline void Pio(Piostream& stream, unsigned long& data) { stream.io(data); }
inline void Pio(Piostream& stream, long long& data) { stream.io(data); }
inline void Pio(Piostream& stream, unsigned long long& data) { stream.io(data); }
inline void Pio(Piostream& stream, double& data) { stream.io(data); }
inline void Pio(Piostream& stream, float& data) { stream.io(data); }
inline void Pio(Piostream& stream, string& data) { stream.io(data); }
inline void Pio(Piostream& stream, Persistent& data) { data.io(stream); }



} // End namespace SCIRun

#endif
