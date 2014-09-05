//static char *id="@(#) $Id$";

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

#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <fstream>
using std::ifstream;
#include <sstream>
using std::istringstream;

namespace SCICore {
namespace PersistentSpace {

static Piostream::MapClStringPersistentTypeID* table = 0;  

//----------------------------------------------------------------------
PersistentTypeID::PersistentTypeID(char* type, char* parent,
  Persistent* (*maker)()):
  type(type), parent(parent), maker(maker)
{

  if (!table) {
    table = scinew Piostream::MapClStringPersistentTypeID;
  }
  
  clString typestring(type);
  
  Piostream::MapClStringPersistentTypeID::iterator dummy;

  dummy = table->find(typestring);
  if (dummy != table->end()) {
    if ((*dummy).second->maker != maker ||
      clString((*dummy).second->parent) != clString(parent))
      {
 	cerr << "WARNING: duplicate type in Persistent "
	     << "Object Type Database: " << typestring << endl;
      }
  }
  
				// should this be else { ?
  (*table)[typestring] = this;
  
}

//----------------------------------------------------------------------
Persistent::~Persistent()
{
}

//----------------------------------------------------------------------
Piostream::Piostream(Direction dir, int version, const clString &name)
: dir(dir), version(version), err(0), outpointers(0), inpointers(0),
  current_pointer_id(1), file_name(name)
{
}

//----------------------------------------------------------------------
Piostream::~Piostream()
{
}

//----------------------------------------------------------------------
int Piostream::reading()
{
    return dir==Read;
}

//----------------------------------------------------------------------
int Piostream::writing()
{
    return dir==Write;
}

//----------------------------------------------------------------------
int Piostream::error()
{
    return err;
}

//----------------------------------------------------------------------
static PersistentTypeID* find_derived(const clString& classname,
				      const clString& basename)
{
  if (!table) return 0;
  PersistentTypeID* pid;
  
  Piostream::MapClStringPersistentTypeID::iterator iter;
  
  iter = table->find(classname);
  if(iter == table->end()) return 0;
  
  pid = (*iter).second;
  if (!pid->parent) return 0;
  if (basename == pid->parent) return pid;
  if (find_derived(pid->parent, basename)) return pid;
  
  return 0;
}

//----------------------------------------------------------------------
void Piostream::io(Persistent*& data, const PersistentTypeID& pid)
{
  if (err) {
    return;
  }
  if (dir == Read) {
    int have_data;
    int pointer_id;
    data=0;			// In case anything goes wrong...
    emit_pointer(have_data, pointer_id);
    if (have_data) {
				// See what type comes next in the
				// stream.  If it Is a type derived
				// from pid->type, then read it in
				// Otherwise, it is an error...
      clString in_name(peek_class());
      clString want_name(pid.type);
      Persistent* (*maker)() = 0;
      if (in_name == want_name) {
	maker=pid.maker;
      }
      else {
	PersistentTypeID* found_pid = find_derived(in_name, want_name);
	if (found_pid) {
	  maker=found_pid->maker;
	}
      }
      if (!maker) {
	cerr << "Maker not found? (class=" << in_name << ")\n";
	err=1;
	return;
      }
      
				// Make it..
      data=(*maker)();
				// Read it in...
      data->io(*this);
      
				// Insert this pointer in the database
      if (!inpointers) {
	inpointers = scinew MapIntPersistent;
      }
      (*inpointers)[pointer_id] = data;
    } else {
				// Look it up...
      if (pointer_id==0) {
	data=0;
      }
      else {
	MapIntPersistent::iterator initer;
	if (inpointers) initer = inpointers->find(pointer_id);
	if (!inpointers || initer == inpointers->end()) {
	  cerr << "Error - pointer not in file, but should be!\n";
	  err=1;
	  return;
	}
	data = (*initer).second;
      }
    }
  }
  else {			// dir == Write
    int have_data;
    int pointer_id;

    MapPersistentInt::iterator outiter;
    if (outpointers) {
      outiter = outpointers->find(data);
      pointer_id = (*outiter).second;
    }
    
    if (data==0) {
      have_data=0;
      pointer_id=0;
    }
    else if (outpointers && outiter != outpointers->end())
      {
				// Already emitted, pointer id fetched
				// from hashtable
	have_data=0;
      }
    else {
				// Emit it..
      have_data=1;
      pointer_id=current_pointer_id++;
      if (!outpointers) {
				// scinew?
	outpointers = new MapPersistentInt;
      }
      (*outpointers)[data] = pointer_id;
    }
    
    emit_pointer(have_data, pointer_id);
    
    if(have_data) {
      data->io(*this);
    }
    
  }
}

//----------------------------------------------------------------------
Piostream* auto_istream(const clString& filename)
{
  std::ifstream in(filename());
  if (!in) {
    cerr << "file not found: " << filename << endl;
    return 0;
  }
  char hdr[12];
  in.read(hdr, 12);
  if (!in) {
    cerr << "Error reading header of file: " << filename << "\n";
    return 0;
  }
  int version;
  if (!Piostream::readHeader(filename, hdr, 0, version)) {
    cerr << "Error parsing header of file: " << filename << "\n";
    return 0;
  }
  if(version != 1){
    cerr << "Unkown PIO version: " << version << ", found in file: " << filename << '\n';
    return 0;
  }
  char m1=hdr[4];
  char m2=hdr[5];
  char m3=hdr[6];
  if(m1 == 'B' && m2 == 'I' && m3 == 'N'){
    return scinew BinaryPiostream(filename, Piostream::Read);
  } else if(m1 == 'A' && m2 == 'S' && m3 == 'C'){
    return scinew TextPiostream(filename, Piostream::Read);
  } else if(m1 == 'G' && m2 == 'Z' && m3 == 'P'){
    return scinew GunzipPiostream(filename, Piostream::Read);
  } else {
    cerr << filename << " is an unknown type!\n";
    return 0;
  }
}

//----------------------------------------------------------------------
bool Piostream::readHeader(const clString& filename, char* hdr,
  const char* filetype, int& version)
{
  char m1=hdr[0];
  char m2=hdr[1];
  char m3=hdr[2];
  char m4=hdr[3];
  if(m1 != 'S' || m2 != 'C' || m3 != 'I' || m4 != '\n') {
    cerr << filename << " is not a valid SCI file! (magic="
	 << m1 << m2 << m3 << m4 << ")\n";
    return false;
  }
  char v[5];
  v[0]=hdr[8];
  v[1]=hdr[9];
  v[2]=hdr[10];
  v[3]=hdr[11];
  v[4]=0;
  istringstream in(v);
  in >> version;
  if(!in){
    cerr << "Error reading file: " << filename << " (while reading version)" << endl;
    return false;
  }
  if(filetype){
    if(hdr[4] != filetype[0] || hdr[5] != filetype[1] || hdr[6] != filetype[2]){
      cerr << "Wrong filetype: " << filename << endl;
      return false;
    }
  }
  return true;
}

//----------------------------------------------------------------------
int Piostream::begin_class(const char* classname, int current_version)
{
  return begin_class(clString(classname), current_version);
}

} // End namespace PersistentSpace
} // End namespace SCICore

//
// $log$
//
