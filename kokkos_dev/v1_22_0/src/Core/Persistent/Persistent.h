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
 *  Persistent.h: Base class for persistent objects...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Persistent_h
#define SCI_project_Persistent_h 1

#include <sgi_stl_warnings_off.h>
#include <map>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <Core/Util/Assert.h>
#include <Core/share/share.h>

namespace SCIRun {

using std::string;
using std::map;
using std::pair;

class Persistent;

//----------------------------------------------------------------------
struct SCICORESHARE PersistentTypeID {
  string type;
  string parent;
  Persistent* (*maker)();
  PersistentTypeID(const string& type, const string& parent,
		   Persistent* (*maker)());
  ~PersistentTypeID();
};

//----------------------------------------------------------------------
class SCICORESHARE Piostream {
  
public:

  typedef map<Persistent*, int>			MapPersistentInt;
  typedef map<int, Persistent*>			MapIntPersistent;
  typedef map<string, PersistentTypeID*>	MapStringPersistentTypeID;

  enum Direction {
    Read,
    Write
  };
  
protected:
  Piostream(Direction, int, const string & =string(""));
  Direction dir;
  int version;
  int err;
  
  MapPersistentInt* outpointers;
  MapIntPersistent* inpointers;
  
  int current_pointer_id;
  virtual void emit_pointer(int&, int&)=0;
  static bool readHeader(const string& filename, char* hdr,
    const char* type, int& version);
public:
  string file_name;

  virtual ~Piostream();
  virtual string peek_class()=0;
  virtual int begin_class(const string& name, int current_version)=0;
  virtual void end_class()=0;

  void io(Persistent*&, const PersistentTypeID&);

  virtual void begin_cheap_delim()=0;
  virtual void end_cheap_delim()=0;

  virtual void io(bool&)=0;
  virtual void io(char&)=0;
  virtual void io(unsigned char&)=0;
  virtual void io(short&)=0;
  virtual void io(unsigned short&)=0;
  virtual void io(int&)=0;
  virtual void io(unsigned int&)=0;
  virtual void io(long&)=0;
  virtual void io(unsigned long&)=0;
  virtual void io(long long&)=0;
  virtual void io(double&)=0;
  virtual void io(float&)=0;
  virtual void io(string& str)=0;

  int reading();
  int writing();
  int error();

  virtual bool supports_block_io() { return false; }
  virtual void block_io(void*, size_t, size_t) { ASSERTFAIL("unsupported"); }

  friend Piostream* auto_istream(const string& filename);
};

//----------------------------------------------------------------------
class SCICORESHARE Persistent {
public:
  virtual ~Persistent();
  virtual void io(Piostream&)=0;
};

//----------------------------------------------------------------------
SCICORESHARE inline void Pio(Piostream& stream, bool& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, char& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, unsigned char& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, short& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, unsigned short& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, int& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, unsigned int& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, long& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, long long& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, unsigned long& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, double& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, float& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, string& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, Persistent& data) { data.io(stream); }



} // End namespace SCIRun

#endif
