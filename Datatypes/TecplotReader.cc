#include <Datatypes/TecplotReader.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/VectorFieldRG.h>
#include <Datatypes/cfdlibParticleSet.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <fstream.h>
#include <iomanip.h>
#include <strstream.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static Persistent* maker()
{
    return scinew TecplotReader();
}

PersistentTypeID TecplotReader::type_id("TecplotReader",
					"ParticleGridReader", maker);

TecplotReader::~TecplotReader()
{
  for(int i = 0; i < fluids.size(); i++ )
    {
      delete fluids[i];
    }
  
}

TecplotReader::TecplotReader() : xmin(0), xmax(1), ymin(1), ymax(1),
  zmin(0), zmax(1), TwoD( true ), startTime(0),
  endTime(0), increment(0)
{
}

TecplotReader::TecplotReader(const clString& filename, int start,
			     int end, int incr) : xmin(0), xmax(1),
  ymin(1), ymax(1), zmin(0), zmax(1), TwoD( true ), startTime( start ),
  endTime( end ), increment(incr), filename(filename)
{
  readfile();
}

TecplotReader::TecplotReader(const clString& file ) : xmin(0), xmax(1),
  ymin(1), ymax(1), zmin(0), zmax(1), TwoD( true ), startTime(0),
  endTime(0), increment(0), filename(file)
{
  readfile();
}

ParticleGridReader* TecplotReader::clone() const
{
  return scinew TecplotReader();
}

void TecplotReader::SetFile(const clString& file )
{
  filename = file;
}

clString TecplotReader::GetFile()
{
  return filename;
}

int TecplotReader::GetNTimesteps()
{
  return endTime - startTime;
}

MEFluid* TecplotReader::getFluid(int i)
{
  if( i < fluids.size())
    return fluids[i];
  else return 0;
}

int TecplotReader::getNFluids()
{
  return fluids.size();
}

void TecplotReader::GetParticleData(int particleId,
				    int variableId,
				    int fluidId,
				    Array1<float>&  values)
{
  if(  endTime - startTime <= 0 )
    return;

  values.remove_all();
  
  float val;
  int index;
  char num[2];
  int fluidIndex;
  // Begin by finding the root file name;
  clString file = basename(  filename );
  clString path = pathname( filename  );
  char *p = file();
  char n[5];
  char root[ 80 ];
  int i, ii;
  int j = 0;
  int k = 0;
  for( i= 0; i < file.len(); i++ )
    {
      if(isdigit(*p)) n[j++] = *p;
      else root[k++] = *p;
      p++;
    }
  root[k] = '\0';

  for( i = startTime; i <= endTime; i += increment){
    ostrstream ostr;
    ostr.fill('0');
    ostr << path << "/"<< root<< setw(4)<<i;
    
    ifstream in( ostr.str(), ios::in);
    if( !in ) {
      cerr << "Error opening file " << filename << endl;
      return;
    }
    char buf [LINEMAX];
    clString token;

    while( in >> token){
      if( token != "ZONE" ){
	in.getline(buf, LINEMAX);
	continue;
      } else { 
	  if( in.getline(buf, LINEMAX) ){
	    if( isBlock( buf ) ) {
	      continue;
	    } else {
	      ii = find( 'I', buf);
	      if (ii < particleId) return;
	      p = buf;
	      while ( !isdigit(*p) && *p != '\0') p++;
	      if( *p == '\0' ) cerr<<"Error in Particle format\n";
	      num[0] = *p;
	      num[1] = '\0';
	      fluidIndex = atoi(num);
	      if ( fluidIndex != fluidId ) {
		continue;
	      } else {
		if( TwoD ) j = 2;
		else j = 3;
		index = j + variableId;
		for( j = 0; j < particleId; j++ ) {
		  for(k = 0; k < variables.size(); k++) {
		    in >> val;
		  }
		}
		for(k = 0; k < variables.size(); k++){
		  in >> val;
		  if ( index == k )
		    values.add(val);
		}		
		
	      }
	    }
	  }
      }
    }
  }
  //  for( i =0; i < values.size(); i++)
  //  cerr<< values[i]<<", ";
  //cerr<<endl;
  NOT_FINISHED("TecplotReader::GetParticleData");
}


#define TECPLOTREADER_VERSION 1

void TecplotReader::io(Piostream& stream)
{
  stream.begin_class("TecplotReader", TECPLOTREADER_VERSION);
  ParticleGridReader::io(stream);

  stream.end_class();
}


int TecplotReader::readfile()
{
  //NOT_FINISHED("TecplotReader::readfile");

  ifstream in;
  in.open( filename(), ios::in );
  if( !in ) {
    cerr << "Error opening file " << filename << endl;
    return 0;
  }

  char buf [LINEMAX];
  clString token;


  while( in >> token ){
    if ( token == "TITLE" ) {
      in.getline(buf, LINEMAX);
      // ignore title
      cout << token <<endl;
    } else if ( token == "TEXT" ) {
      in.getline(buf, LINEMAX);
      // ignore text
      cout << token <<endl;
    } else if ( token == "T") {
      in.getline(buf, LINEMAX);
      // ignore more text
      cout << token <<endl;
    } else if ( token == "VARIABLES" ) {
      cout << token <<endl;
      readVars(in);
    } else if ( token == "ZONE" ){
      readZone(in);
    }
  }
  in.close();
  return 1;
}

void TecplotReader::readVars(istream& is)
{
  char c;
  char buf[LINEMAX];
  char tok[VARNAMELEN];
  clString token;
  int nFluids = 1;

  while( is.getline(buf,LINEMAX) ){
    int readin = is.gcount() - 1;
    if( readin < LINEMAX ) {
      buf[readin++] = '\n';	//  place newline back in string
      buf[readin] = '\0';       //  append a null character
    }

    istrstream iss(buf);
    
    while( c = iss.get() ) {  // remove whitespace and =
      if ( c != ' ' && c != '\t' && c != '\n' && c != '=')
	break;
    }
    
    iss.putback(c);
    while( iss.good()) {

      char *p = tok;
      while (c = iss.get()){
	if( c != ',' && c != '\n') {
	  *p = c; 
	  p++;
	}
	else break;
      }
      *p = '\0';		// Put names in tok
      
      token = tok;
      if(token == "Z")  TwoD = false;
      if (token != "" ){
	variables.add(token);
	setFluidNum( token, nFluids );
      }
      
      if ( c == '\n' ) {	// The last character is a newline
	break;			// we're done reading.
      }
      c = iss.peek();		// If we have made it here
      if( c == '\n' ){          // the previous char was a comma and
	c = ',';		//  now we have a newline.  We must 
        break;                  // read another line of variables.
      }         		
    }
    if ( c == '\n' ) break;     // we're done
  }
  int i;
  for(i = 0; i < variables.size(); i++)
    cout << variables[i] << " ";
  cout << endl << "nvars = "<< variables.size()<< endl;
  cout << "nFluids = "<<nFluids<<endl;

				//  Now create fluids
  fluids.remove_all();

  for(i = 0; i < nFluids; i++)
    {
      fluids.add( scinew MEFluid() );
   }
	
}

void removeChars(istream& is)
{
  char c;
  while ( c = is.get() ) {
    if( c != '"' && c != ' ' && c != '=' && c != ',')
      break;
  }
}

int TecplotReader::find( char c, char *buf)
{
  istrstream iss(buf);
  char tok[LINEMAX];
  int i;
  iss.get(tok,LINEMAX,',');
  iss.get(); //remove comma
  if( iss.get(tok, LINEMAX, c) )
    {
      //      cerr<< tok << "\t";
      if( iss.get(tok, LINEMAX, '=') ) {
	iss.get(); // remove =
	
	if( iss.get(tok, LINEMAX, ',')) {
	  clString str(tok);
	  str.get_int(i);
	  //	  cerr << c <<" = "<<i<<endl;
	  return i;
	} else return 0;
      } else return 0;
    } else return 0;
}

int TecplotReader::isBlock( char *buf)
{
  if( find( 'I', buf ) && find('J', buf) )
    return 1;
  else
    return 0;
}
	 
  
  
void TecplotReader::readZone(istream& is)
{
  char *p;
  char buf[LINEMAX];
  char num[2];
  int i = 1, j = 1, k = 1, fluidIndex = 1; // index is the fluid index
  
  if( is.getline(buf, LINEMAX) ){
    if( isBlock( buf ) )
      {
	i = find( 'I', buf );
	j = find( 'J', buf );
	k = find( 'K', buf );

        k = ( (k == 0) ? 1 : k);
	
	readBlock( i, j, k, is);
      } else {
	i = find( 'I', buf);
	p = buf;
	while ( !isdigit(*p) && *p != '\0') p++;
	if( *p == '\0' ) cerr<<"Error in Particle format\n";
	num[0] = *p;
	num[1] = '\0';
	fluidIndex = atoi(num) -1;
	readParticles( i, fluidIndex, is);
      }
   }
}

void TecplotReader::readParticles(int ii, int fluidIndex, istream& is)
{
  // Each particle variable is read in order.  e.g. for particle 1
  // read all variables, for particle 2 read all variables, etc.
  // Unfortuately the cfdlibParticleSet stores variables in arrays
  // that correspond to a position array--- all positions are in 
  // one big array, all Ts are in one array, all UVWs, etc.
  // To add to the confusion a particle only corresponds to 1 fluid
  // even though the data file contains data points that correspond
  // to all fluids.  Let's say that we have variables X,Y,P,T1,RO1,U1,V1
  // T2,RO2,U2,V2.  In this case every value beyond V1 is thrown away 
  // and should only contain zeros anyhow.  I have already determined
  // which fluid this particle corresponds to and have passed that info
  // in via the fluidIndex variable.
  int i, v, index = 0;
  clString var, stripped;
  cfdlibParticleSet *ps = scinew cfdlibParticleSet();
  cfdlibTimeStep* ts = scinew cfdlibTimeStep();
  

  // First lets walk through the variables and add them to the 
  // particle set, ignoring any variables beyond the "1" vars.
  // Also we must index the variables that correspond to vectors. 
  // Generally this is position, direction and momentum. Finally
  // We keep an index to the last "1" variable so we know we can 
  // throw away anthing beyond that.
  Array1< int > vids;
  Array1< int > sids;
  int lastID = 0;
  for( v = 0; v < variables.size(); v++) {

    var = variables[v];
    char *p = var();
    stripVar( var, stripped, index );
    if (index > 1 ) continue;  // we don't care
    else {
      if( *p == 'X'){
	vids.add(v);   // a vectorvar index
	if( TwoD ) {
	  v++;
	  ps->addVectorVar( clString( "XY" ) );
	} else {
	  v += 2;
	  ps->addVectorVar( clString( "XYZ" ) );
	}
      } else if( *p == 'U' ) {
	vids.add(v);  // a vectorvar index
	if( TwoD ) {
	  v++;
	  ps->addVectorVar( clString( "UV" ) );
	} else {
	  v += 2;
	  ps->addVectorVar( clString( "UVW" ) );
	}
      } else if( *p == 'M') {
	vids.add(v);  // a vectorvar index
	if( TwoD ) {
	  v++;
	  ps->addVectorVar( clString( "MOXY" ) );
	} else {
	  v += 2;
	  ps->addVectorVar( clString( "MOXYZ" ) );
	}
      } else {   // a scalar variable
	ps->addScalarVar( stripped );
	sids.add(v);
      }
      lastID = v;  // get pointer to the last "good" var index.
    }
  }

  // Now we know where our vectors are, and the last good index. 
  // We also know how many scalar and vector variables we have.
  // Let's resize the timestep array.
  ts->vectors.resize( vids.size() );
  ts->scalars.resize( sids.size() );
 
  float x, y, z;

  for(i = 0 ; i < ii; i++ ){
    int vid = 0;  
    int sid = 0;

    for(v = 0; v < variables.size(); v++) {
      if ( v > lastID ){
	is >> x;
      } else {
	if (v == vids[vid]){ // we have a vector variable
	  if( TwoD ) {
	    is >> x >> y;
	    ts->vectors[vid].add( Vector( x, y, 0) );
	    v++;
	  } else {
	    is >> x >> y >> z;
	    ts->vectors[vid].add( Vector( x, y, z) );
	    v+=2;
	  }
	  vid++; 
	} else {  // we have a scalar variable
	  is >> x;
	  ts->scalars[sid].add( x );
	  sid++;
	}
      }
    }
  }

  // Add the timestep to the particle set
  ps->add( ts );

  ParticleSetHandle psh = ParticleSetHandle( ps );
  fluids[fluidIndex]->AddParticleSet( psh );
}

void TecplotReader::readBlock(int i, int j, int k, istream& is)
{
  char *p;
  int index;  // Note that for the ME guys indexing starts at 1.
  clString stripped;


  cerr<<"Working on Block of size "<< i << " by "<<j<<" by "<< k <<endl;

  for(int v = 0; v < variables.size(); v++)
    {
      clString var = variables[v];
      //      cerr << var <<endl;
      if( var == "X"){ // || variables[v] == "Y") {
	getBounds(xmin, xmax, i, j, k, is);
      } else if(var == "Y") {
	getBounds(ymin, ymax, i, j, k, is);
      } else if(var == "Z") {
	getBounds(zmin, zmax, i, j, k, is);
      } else if(var == "P") {
	ScalarFieldHandle sfh = makeScalarField(i,j,k,is);
	for(int f=0; f < fluids.size(); f++)
	  fluids[f]->AddScalarField( var, sfh );
      }	else if ( *(p = var()) == 'U' ){
				//  We can assume the next two (Possibly 
				//  three) variables make up the vectorfield
	VectorFieldHandle vfh = makeVectorField(i,j,k,is);
	stripVar(var,  stripped, index );
	if( TwoD ) {
	  v ++;
	  fluids[index-1]->AddVectorField( clString("UV"), vfh);
	} else {
	  v += 2;
	  fluids[index-1]->AddVectorField( clString("UVW"), vfh);
	}
      } else if ( *(p = var()) == 'M') { // Momentum
	VectorFieldHandle vfh = makeVectorField(i,j,k,is);
	stripVar(var,  stripped, index );
	if( TwoD ) {
	  v ++;
	  fluids[index-1]->AddVectorField( clString("MOXY"), vfh);
	} else {
	  v += 2;
	  fluids[index-1]->AddVectorField( clString("MOXYZ"), vfh);
	}
      } else {
	ScalarFieldHandle sfh = makeScalarField(i,j,k, is);
	stripVar(var, stripped, index);
	fluids[index-1]->AddScalarField( stripped, sfh);
      }
	
    }      
  
  NOT_FINISHED("TecplotReader::readBlock");
}


void TecplotReader::getBounds(double& min, double& max,
			      int ii, int jj, int kk,
			      istream& is)
{
  int i,j,k;
  double val;
  min= 1.0E+50;
  max= -1.0E+50;
  for(k = 0; k < kk; k++)
    for(j = 0; j < jj; j++)
      for(i = 0; i < ii; i++){
	is >> val;
	max = ( max < val ) ? val : max;
	min = ( min > val ) ? val : min;
      }
}


void TecplotReader::setFluidNum( clString str, int& nFluids)
{
  for(int i = 2; i < MAXFLUIDS; i++){
    clString n(to_string(i));
    char *p = n();
    if ( strchr(str(), p[0]) != NULL && nFluids < i){
      nFluids = i;
    }
  }
}


void TecplotReader::stripVar( const clString& var, clString& retVar, int& index )
{
  char *p = var();
  char v[VARNAMELEN];
  int vindex = 0;
  for (int i = 0; i < var.len(); i++)
    {
      if (isdigit( *p )){
	char num[2];
	num[0] = *p;
	num[1] = '\0';
	index = atoi( num );
	p++;
      }
      else {
	v[vindex++] = *p;
	p++;
      }
    }
  v[vindex] = '\0';
  retVar = v;
}
  
// ScalarFieldHandle TecplotReader::makeScalarField(int, int, int, istream&)
ScalarFieldHandle TecplotReader::makeScalarField(int ii, int jj,
						 int kk, istream& is)
{

  ScalarFieldRG *sf = scinew ScalarFieldRG();
  int i, j, k;
  float s;

  if( TwoD ) {  // zmin = zmax.  We must make the field 3D.
    sf->resize(ii,jj,2);
    sf->set_bounds(Point(xmin,ymin,zmin), Point(xmax,ymax,zmax+1));
  } else {
    sf->resize(ii,jj,kk);
    sf->set_bounds(Point(xmin,ymin,zmin), Point(xmax,ymax,zmax));
  }

  for(k = 0; k < kk; k++){
    for(j = 0; j < jj; j++){
      for(i = 0; i < ii; i++ ) {
	is >> s;
	sf->grid(i,j,k) = s;
	if( TwoD ){
	  sf->grid(i,j,k+1) = s;
	}
      }
    }
  }
  return ScalarFieldHandle( sf );
}

//VectorFieldHandle TecplotReader::makeVectorField(int,int,int,istream&)
VectorFieldHandle TecplotReader::makeVectorField(int ii, int jj,
						 int kk, istream& is)
{
  VectorFieldRG *vf = scinew VectorFieldRG();
  
  int i,j,k;
  float x,y,z;

  if( TwoD ) {  // zmin = zmax.  We must make the field 3D.
    vf->resize(ii,jj,2);
    vf->set_bounds(Point(xmin,ymin,zmin), Point(xmax,ymax,zmax+1));
  } else {
    vf->resize(ii,jj,kk);
    vf->set_bounds(Point(xmin,ymin,zmin), Point(xmax,ymax,zmax));
  }
    

  for(k = 0; k < kk; k++){  //read all the the first var
    for(j = 0; j < jj; j++){
      for(i = 0; i < ii; i++ ) {
	is >> x;
	vf->grid(i,j,k) = Vector(x,0,0);
      }
    }
  }

  for(k = 0; k < kk; k++){ // read the second var
    for(j = 0; j < jj; j++){
      for(i = 0; i < ii; i++ ) {
	is >> y;  
	vf->grid(i,j,k).y(y);
      }
    }
  }

  if ( TwoD ) { // no more reading
    for(j = 0; j < jj; j++){
      for(i = 0; i < ii; i++ ) {
	vf->grid(i,j,1) = vf->grid(i,j,0);
      }
    }
  } else {
    for(k = 0; k < kk; k++){
      for(j = 0; j < jj; j++){
	for(i = 0; i < ii; i++ ) {
	  is >> z;
	  vf->grid(i,j,k).z(z);
	}
      }
    }
  }
  return VectorFieldHandle( vf );
}

