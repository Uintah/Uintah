//////////////////////////////////////////////////////////////////////
// SaveVRML.h - Write a vector<Object> as a VRML 1.0 file.
// Copyright David K. McAllister July 1999.
//////////////////////////////////////////////////////////////////////

#ifndef _wrobject_h
#define _wrobject_h

#include <Packages/Remote/Tools/Model/Model.h>

#include <stdio.h>

namespace Remote {
class WrObject : public Object
{
  FILE *out;
  int Ind;

  inline void IncIndent()
  {
    Ind += 2;
  }

  inline void DecIndent()
  {
    Ind -= 2;
  }

  void indent();
  void writeVector(const Vector &);
  void writeMaterials();
  void writeNormals();
  void writeTexCoords();
  void writeVertices();
  void writeIndices();
public:
  inline WrObject()
  {
    Ind = 0;
    out = NULL;
  }

  inline WrObject(Object o) : Object(o)
  {
    Ind = 0;
    out = NULL;
  }

  void Write(FILE *, int indlevel);
};

} // End namespace Remote


#endif
