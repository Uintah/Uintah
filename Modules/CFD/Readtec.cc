/*
 *  Readtec.cc: Read in tecplot file and create scalar field, vector
 *             field, and particle set.
 *
 *  Written by:
 *    Philip Sutton
 *   Department of Computer Science
 *   University of Utah
 *   May 1998
 *
 *  Copyright (C) 1999 SCI Group
 */
  

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Classlib/Array1.h>
#include <Classlib/Array2.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/VectorFieldRG.h>
#include <Datatypes/VectorFieldPort.h>
#include <Datatypes/cfdlibParticleSet.h>
#include <Datatypes/ParticleSetPort.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>
#include <stdio.h>
#include <string.h>
#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>
#include <ctype.h>
#include <Classlib/String.h>
#include <Classlib/Assert.h>


// A Zone contains an array with all the variable values, index by
// variable number
struct Zone {
  int i, j;
  Array2<double> varvals;
};



class Readtec : public Module {
private:
  // CONSTANTS
  // max line length
  static const int LINEMAX = 1000;
  // 4(X,Y,Z,P) +  4(max # fluids) * 9(fluid variables)
  static const int MAXVARS = 40;
  // max length of a variable name
  static const int VARNAMELEN = 40;
  
private:
  // PRIVATE VARIABLES
  TCLstring filebase;
  ScalarFieldOPort *sfout;
  VectorFieldOPort *vfout;
  ParticleSetOPort *psout;
  
  // actual number of scalar variables parsed
  int numvars;
  // number of fluids in the dataset
  int numfluids;
  // list of variable names, indexed by number
  Array1<char*> varnames;
  // list of zones
  Array1<Zone> zonelist;
  // the regular grid for scalars (always that way for tecplot?)
  ScalarFieldRG *sfgrid;
  // the regular grid for vectors
  VectorFieldRG *vfgrid;
  // particle set
  cfdlibParticleSet *pset;
  
  // used for visualizing different variables
  TCLint sfluid, svar;
  TCLint vfluid, vvar;
  TCLint pfluid, pvar;
  char svartable[6][VARNAMELEN];
  
  // tells us if we have to read the file or not
  char lastfileread[80];
  
private:
  // PRIVATE METHODS
  void gettoken( ifstream &in, char *buf, int len );
  int readfile( char *filename, int *index );
  void CreateSFgrid( Zone *zone );
  void CreateVFgrid( Zone *zone );
  void CreatePS( Zone *zone, int I, int J );
  void initvars();
  void checkfluidnum( char *name);
public:
  Readtec(const clString& id);
  Readtec(const Readtec&, int deep);
  virtual ~Readtec();
  virtual Module* clone(int deep);
  virtual void execute();
};

extern "C" {
Module* make_Readtec(const clString& id)
{
    return new Readtec(id);
}
}

Readtec::Readtec(const clString& id)
: Module("Readtec", id, Filter), filebase("filebase", id, this),
  svar("svar", id, this), sfluid("sfluid", id, this), vvar("vvar", id, this),
  vfluid("vfluid", id, this ), pfluid("pfluid", id, this), pvar("pvar",id,this)
{
    // Create the output port
    sfout=new ScalarFieldOPort(this, "ScalarField", ScalarFieldIPort::Atomic);
    vfout=new VectorFieldOPort(this, "VectorField", VectorFieldIPort::Atomic);
    psout=new ParticleSetOPort(this, "ParticleSet", ParticleSetIPort::Atomic);
    add_oport(sfout);
    add_oport(vfout);
    add_oport(psout);
    initvars();
}

Readtec::Readtec(const Readtec& copy, int deep)
: Module(copy, deep), filebase("filebase", id, this),
  svar("svar", id, this), sfluid("sfluid", id, this), vvar("vvar", id, this),
  vfluid("vfluid",id, this ), pfluid("pfluid", id, this), pvar("pvar",id,this)
{
  initvars();
}

Readtec::~Readtec()
{
  initvars();
}

Module* Readtec::clone(int deep)
{
    return new Readtec(*this, deep);
}

int Readtec::readfile( char *filename, int *index ) {
  ifstream in;
  in.open( filename, ios::in );
  if( !in ) {
    cerr << "Error opening file " << filename << endl;
    return 1;
  }

  char buf[LINEMAX];
  char tok[LINEMAX];
  
  // read each line until EOF
  while( in.getline( buf, LINEMAX ) ) {

    // read in first token
    int n = sscanf(buf, "%s", tok );
    if( n != 1 )
      continue; // blank line - do nothing

    if( !strcmp( tok, "TITLE" ) ) {
      // ignore title
    } else if( !strcmp( tok, "TEXT" ) ) {
      // ignore text
    } else if( !strcmp( tok, "T" ) ) {
      // more text = ignore
    } else if( !strcmp( tok, "VARIABLES" ) ) {
      int i = 0;

      // get rid of " = "
      char *p = buf;
      while( *p != '=' && *p != 0 ) p++;
      p++;
      while( isspace(*p) && *p != 0 ) p++;

      int getnext = 0;
      int done = 0;
      // parse variable names
      char *name;
      while( !done ) {
	name = new char[VARNAMELEN];
	int j;
	
	for(j = 0; !isspace(*p) && *p != 0 && *p != ','; j++, p++ ) 
	  name[j] = *p;
	ASSERT(j < VARNAMELEN-1);
	for( ; j < VARNAMELEN; j++ )
	  name[j] = '\0';
	checkfluidnum( name );
	if( *p == ',' ) {  p++;  getnext = 1; }
	while( isspace(*p) & *p != 0 ) p++;

	// hit end of line while still looking for variables - read in another
	if( *p == 0 && getnext ) {
	  in.getline( buf, LINEMAX );
	  p = buf;
	  while( isspace(*p) && *p != 0 ) p++;
	} else if( *p == 0 ) {
	  done = 1;
	} else {
	  getnext = 0;
	}

	varnames.add(name);
	i++;
	ASSERT( i < MAXVARS );
      }
      numvars = i;
//    for displaying variable list
//      int j;
//      for(j = 0; j < numvars; j++ ) 
//	cout << varnames[j] << endl;

    } else if( !strcmp( tok, "ZONE" ) ) {
      char *p = buf;
      int I = 0, J = 1, k;
      int block;
      char num[LINEMAX];
      char title[LINEMAX];

      // this is so we can display which zone is currently being parsed,
      // since it tends to take quite a while for large zones.
      for( k=0; buf[k] != '\0'; k++ )
	title[k] = buf[k];
      ASSERT(k<LINEMAX-1);
      for( ; k < LINEMAX; k++ )
	title[k] = '\0';
      
      // get rid of title
      while( *p != '"' && *p != 0 ) p++;
      p++;
      while( *p != '"' && *p != 0 ) p++;
      p++;

      // now find I
      while( *p != 'I' && *p != 0 ) p++;
      p++;
      while( (isspace( *p ) || *p == '=' ) && *p != 0 ) p++;
      for( k = 0; *p != ',' && *p != 0; p++, k++ )
	num[k] = *p;
      ASSERT(k<LINEMAX);
      I = atoi(num);

      // is J or F next? if J it's a block, if F it's a point
      while( ( isspace( *p ) || *p == ',' ) && *p != 0 ) p++; 
      if( *p == 'J' ) {
	// find J
	while( *p != 'J' && *p != 0 ) p++;
	p++;
	while( (isspace( *p ) || *p == '=' ) && *p != 0 ) p++;
	for( k = 0; *p != ',' && *p != 0; p++, k++ )
	  num[k] = *p;
	ASSERT(k<LINEMAX);
	J = atoi(num);
	block = 1;
      } else if( *p == 'F' ) {
	block = 0;
      }
      // ignore the rest of the line - values start on the next line
      
      Zone zone1;
      zone1.varvals.newsize(MAXVARS,I);

      // parse the variables into the zone structure
      cout << "Working on " << title << " . . ." << endl;
      int zindex;
      if( block ) {

	zone1.varvals.newsize(MAXVARS,I*J);
	// in a block, all values for a variable are listed together
	for( zindex = 0; zindex < numvars; zindex++ ) {
	  int i, j;
	  
	  // read in all the values
	  for( j = 0; j < J; j++ ) {
	    for ( i = 0; i < I; i++ ) {
	      char ch;
	      while( isspace( in.peek() ) ) in.get(ch);
	      gettoken( in, buf, LINEMAX );
	      zone1.varvals(zindex,j*I+i) = atof(buf);
	    }
	  }
	  zone1.i = I;
	  zone1.j = J;
	  
	} // end for zindex
	*index = zonelist.size();
      } else {

	int i;
	// go through each point
	for( i = 0; i < I; i++ ) {
	  char ch;
	  // go through each variable
	  for( zindex = 0; zindex < numvars; zindex++ ) {

	    while( isspace( in.peek() ) ) in.get(ch);
	    gettoken( in, buf, LINEMAX );

	    zone1.varvals(zindex,i) = atof(buf);
	  }
	}
	zone1.i = I;
	zone1.j = 1;
	
      } // end if block

      zonelist.add( zone1 );
      cout << "Done." << endl;

    } // end if !strcmp( tok, "ZONE" )

  } // end while in.getline
  return 0;
}  

void Readtec::execute()
{
  char filename[255];
  static int zoneindex=-1;

  // might have multiple filenames later for animations
  sprintf( filename, "%s", filebase.get()() );
  if( !strcmp( filename, "" ) )
    return;
  
  if( strcmp( filename, lastfileread ) ) {
    // new file - open and read.  Return if hit an error.
    if( readfile( filename, &zoneindex ) )
      return;
    strcpy( lastfileread, filename );
  }    

  // set up the grids
  Zone z = zonelist[zoneindex];
  sfgrid = new ScalarFieldRG();
  sfgrid->resize(z.i,z.j,2);
  CreateSFgrid( &z );

  sfgrid->set_bounds(Point(0,0,0), Point(z.i, z.j, z.i) );
  sfgrid->compute_minmax();

  vfgrid = new VectorFieldRG();
  vfgrid->resize(z.i,z.j,2);
  CreateVFgrid( &z );
  vfgrid->set_bounds(Point(0,0,0), Point(z.i,z.j,z.i) );

  pset = new cfdlibParticleSet();
  if( pfluid.get() - 1 >= zonelist.size() ) {
    cerr << "Error - particle set/fluid " << pfluid.get()
	 << " does not exist" << endl;
  } else {
    Zone z1 = zonelist[ pfluid.get() - 1 ];
    CreatePS( &z1, z.i, z.j );
  }

  sfout->send( ScalarFieldHandle(sfgrid) );
  vfout->send( VectorFieldHandle(vfgrid) );
  psout->send( ParticleSetHandle(pset) );
}

// fills buf with the next token (space separated).
// necessary because CC's implementation of iostream::getline ignores
// the 'delimiter' argument and always uses newline as the delimiter.
void Readtec::gettoken( ifstream &in, char *buf, int len ) {
  int idx;
  for( idx=0; idx < len && !isspace( in.peek() ) ; idx++ )
    in.get( buf[idx] );
  ASSERT(idx<len-1);
  for( ; idx < len; idx++ )
    buf[idx] = '\0';
}

void Readtec::initvars() {
  sprintf(svartable[0],"P");
  sprintf(svartable[1],"T");
  sprintf(svartable[2],"E");
  sprintf(svartable[3],"RHO");
  sprintf(svartable[4],"CKS");
  sprintf(svartable[5],"THE");

  strcpy( lastfileread, "" );
  numfluids = 1;
  svar.set(0);
  vvar.set(0);
  sfluid.set(1);
  vfluid.set(1);
  pfluid.set(1);
  pvar.set(0);
}

void Readtec::checkfluidnum( char *name ) {
  if( strchr( name, '2' ) != NULL && numfluids < 2 )
    numfluids = 2;
  else if( strchr( name, '3' ) != NULL && numfluids < 3 )
    numfluids = 3;
  else if( strchr( name, '4' ) != NULL && numfluids < 4 )
    numfluids = 4;    
}

void Readtec::CreateSFgrid( Zone *zone ) {
  char vname[VARNAMELEN];
  int svnum = svar.get();
  int i;

  if( numfluids == 1 )
    sprintf( vname, "%s", svartable[svnum] );
  else if( svnum == 0 ) 
    sprintf( vname, "P" );
  else 
    sprintf( vname, "%s%d", svartable[svnum], sfluid.get() );  

  for( i = 0; i < numvars && strcmp(varnames[i],vname); i++ );
  if( i == numvars ) {
    fprintf( stderr, "Error - variable %s not found\n", vname );
    return;
  }
  int x,y;
  for( y = 0; y < zone->j; y++ ) {
    for( x = 0; x < zone->i; x++ ) {
      sfgrid->grid(x,y,0) = zone->varvals(i, y*zone->i + x);
      sfgrid->grid(x,y,1) = zone->varvals(i, y*zone->i + x);
    }
  }

}

void Readtec::CreateVFgrid( Zone *zone ) {
  char vname1[VARNAMELEN];
  char vname2[VARNAMELEN];
  int vnum = vvar.get();
  
  switch( vnum ) {
  case 0: sprintf( vname1, "U%d", vfluid.get() );
    sprintf( vname2, "V%d", vfluid.get() ); break;
  case 1: sprintf( vname1, "MO%dX", vfluid.get() );
    sprintf( vname2, "MO%dY", vfluid.get() ); break;
  default:
    fprintf( stderr, "Error - vector variable #%d for fluid %d not found",
	     vnum, vfluid.get() );
    return;
  }

  int i, j;
  for( i = 0; i < numvars && strcmp(varnames[i],vname1); i++ );
  for( j = 0; j < numvars && strcmp(varnames[j],vname2); j++ );
  if( i == 0 || j == 0 ) {
    fprintf( stderr, "Error - one or both of %s, %s not found\n", vname1,
	     vname2 );
    return;
  }

  int x, y;
  for( y = 0; y < zone->j; y++ ) {
    for( x = 0; x < zone->i; x++ ) {
      vfgrid->grid(x,y,0) = Vector( zone->varvals(i, y*zone->i + x),
				    zone->varvals(j, y*zone->i + x),0 );
      vfgrid->grid(x,y,1) = Vector( zone->varvals(i, y*zone->i + x),
				    zone->varvals(j, y*zone->i + x),0 );
    }
  }
      
}

void Readtec::CreatePS( Zone *zone, int I, int J ) {
  cfdlibTimeStep *ts = new cfdlibTimeStep;
  ts->time = 0.0;
  int i;

  // find indices of the X and Y variables
  int xindex;
  for( xindex = 0; xindex < MAXVARS && strcmp( varnames[xindex], "X" );
       xindex++ );
  ASSERT(xindex < MAXVARS);
  int yindex;
  for( yindex = 0; yindex < MAXVARS && strcmp( varnames[yindex], "Y" );
       yindex++ );
  ASSERT(yindex < MAXVARS);

  // find variable name
  char varname[VARNAMELEN];
  if( numfluids == 1 )
    sprintf( varname, svartable[ pvar.get() ] );
  else if( pvar.get() == 0 )
    sprintf( varname, "P" );
  else
    sprintf( varname, "%s%d", svartable[ pvar.get() ], pfluid.get() );

  // find variable in zone
  int varindex;
  for( varindex = 0; varindex < MAXVARS &&
	 strcmp( varnames[varindex], varname ); varindex++ );
  ASSERT( varindex < MAXVARS );

  ts->positions.resize( zone->i );
  ts->scalars.resize( zone->i );
  for( i = 0; i < zone->i; i++ ) {
    ts->positions[i] = Vector( (double)I*zone->varvals(xindex,i)/(double)I,
			       (double)J*zone->varvals(yindex,i)/(double)J,
    			       (double)I/2.0 );
    ts->scalars[i] = zone->varvals( varindex, i );
  }
  pset->add(ts);
}
