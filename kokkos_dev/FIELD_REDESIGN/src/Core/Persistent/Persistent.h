
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

#include <map.h>
#include <iosfwd>

#include <SCICore/Containers/String.h>
#include <SCICore/share/share.h>

namespace SCICore {
  namespace Containers {
    class clString;
  }
}

namespace SCICore {
namespace PersistentSpace {

using SCICore::Containers::clString;

class Persistent;

//----------------------------------------------------------------------
struct SCICORESHARE PersistentTypeID {
  char* type;
  char* parent;
  Persistent* (*maker)();
  PersistentTypeID(char* type, char* parent, Persistent* (*maker)());
};

//----------------------------------------------------------------------
class SCICORESHARE Piostream {
  
public:

  typedef map<Persistent*, int>			MapPersistentInt;
  typedef map<int, Persistent*>			MapIntPersistent;
  typedef map<clString, PersistentTypeID*>	MapClStringPersistentTypeID;

  enum Direction {
    Read,
    Write
  };
  
protected:
  Piostream(Direction, int, const clString & =clString(""));
  Direction dir;
  int version;
  int err;
  
  MapPersistentInt* outpointers;
  MapIntPersistent* inpointers;
  
  int current_pointer_id;
  virtual void emit_pointer(int&, int&)=0;
  static bool readHeader(const clString& filename, char* hdr,
    const char* type, int& version);
public:
  clString file_name;

  virtual ~Piostream();
  virtual clString peek_class()=0;
  int begin_class(const char* name, int current_version);
  virtual int begin_class(const clString& name, int current_version)=0;
  virtual void end_class()=0;

  void io(Persistent*&, const PersistentTypeID&);

  virtual void begin_cheap_delim()=0;
  virtual void end_cheap_delim()=0;

  virtual void io(char&)=0;
  virtual void io(unsigned char&)=0;
  virtual void io(short&)=0;
  virtual void io(unsigned short&)=0;
  virtual void io(int&)=0;
  virtual void io(unsigned int&)=0;
  virtual void io(long&)=0;
  virtual void io(unsigned long&)=0;
  virtual void io(double&)=0;
  virtual void io(float&)=0;
  virtual void io(clString& string)=0;

  int reading();
  int writing();
  int error();

  friend Piostream* auto_istream(const clString& filename);
};

//----------------------------------------------------------------------
class SCICORESHARE Persistent {
public:
  virtual ~Persistent();
  virtual void io(Piostream&)=0;
};

//----------------------------------------------------------------------
SCICORESHARE inline void Pio(Piostream& stream, char& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, unsigned char& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, short& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, unsigned short& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, int& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, unsigned int& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, long& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, unsigned long& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, double& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, float& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, clString& data) { stream.io(data); }
SCICORESHARE inline void Pio(Piostream& stream, Persistent& data) { data.io(stream); }

				// persistent io for maps
template <class Key, class Data>
SCICORESHARE void
  Pio(Piostream& stream, map<Key, Data>& data );


} // End namespace PersistentSpace
} // End namespace SCICore

namespace SCICore {
  namespace Containers {
    
    //typedef Persistent* PersistentPointer;
    
/*
//const??
//----------------------------------------------------------------------
inline SCICORESHARE int Hash(PersistentSpace::Persistent * k, int hash_size)
{
  return int( (unsigned long) ((long)k ^ (3 * hash_size + 1)) %
              (unsigned long) hash_size );
}
*/

  }
}


#endif

