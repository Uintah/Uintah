#ifndef _GRIDLINES_H
#define _GRIDLINES_H

#include <SCICore/TclInterface/TCLvar.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Geom/GeomObj.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>

namespace Uintah {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Containers;
using namespace SCICore::Math;


/******************************************
CLASS
   GridLines
     A module that displays gridlines around a scalar field.


GENERAL INFORMATION
   GridLines.h
   Written by:

     Kurt Zimmerman
     Department of Computer Science
     University of Utah
     December 1999

     Copyright (C) 1999 SCI Group

KEYWORDS
   GridLines, ScalarField, VectorField

DESCRIPTION
   This module was created for the Uintah project to display that actual
   structure of the scalar or vector fields that were being used
   during simulation computations.  The number of lines displayed represent
   the actual grid and cannot be manipulated. This module is based on 
   Philip Sutton's cfdGridLines.cc which was based on FieldCage.cc by
   David Weinstein.

***************************************** */



class GridLines : public Module {
public:
  // GROUP: Constructors: 
  ////////// 
  // Contructor taking 
  // [in] string as an identifier 
  GridLines(const clString& id);

  // GROUP: Destructors: 
  ////////// 
  // Destructor
  virtual ~GridLines();

  ////////// 
  // execution scheduled by scheduler   
   virtual void execute();

private:
  ScalarFieldIPort* insfield;
  VectorFieldIPort* invfield;
  GeometryOPort* ogeom;
  MaterialHandle matl;
  MaterialHandle white;
  TCLdouble rad;
  TCLint mode;
  TCLint textSpace;
  TCLint lineRep;
  TCLint dim;
  TCLint plane;
  TCLdouble planeLoc;
};

  //Should be moved elsewhere
class GeomLineFactory
{
public:
  static GeomObj* Create( int id , const Point&, const Point&,
			   double rad = 0 , int nu = 20, int nv = 1);
};

} // end namespace Modules
} // end namespace Uintah
#endif
