//static char *id="@(#) $Id$";

/****************************************
CLASS
    MPRead

    A class for reeding material particle files.

OVERVIEW TEXT


KEYWORDS
    Material/Particle Method

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    June 1999

    Copyright (C) 1999 SCI Group

LOG
    Created June 28, 1999
****************************************/

#define DEFINE_OLD_IOSTREAM_OPERATORS
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldRGCC.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Datatypes/VectorFieldRGCC.h>
#include "MPRead.h"

#include <sstream>
using namespace std;

namespace Uintah {
namespace Datatypes {  

using namespace SCICore::Datatypes;


MPRead::MPRead(clString fname) :is(fname(), ios::in), filename(fname),
  state(Closed), headerRead( false ), pCount(0),
  svCount(0), vvCount(0), minPt(0,0,0), maxPt(0,0,0)
{
  if (!(is.good())) {
      std::cerr<<"Error in opening file: "<<fname<<endl;
  } else {
    state = Open;
  }
  
  
}

MPRead::~MPRead()
{
  is.close();

}
     
int 
MPRead::ReadHeader( clString& title,
		    clString& filetype,
		    clString& comments)
{  
  if(!(state == Open)){
    cerr<<"Error:  file is not open\n";
    return 0;
  } else {
    state = Header;
    clString in;
    is >> in;
    const int buflen = 1024;
    char buf[buflen];
    if( in != "MPD" ){
      cerr<<"Error: not a material/particle data file\n";
      return 0;
    } else {
      is >> version;
      is >> filetype;
      if(filetype == "BIN") this->fileType = BIN;
      else this->fileType = ASCII;
      while( is >> in ){
	if( in == "TITLE:"){
	  is.getline(buf, buflen);
	  title = clString(buf);
	}
	else if( in == "##"){
	  is.getline(buf, buflen);
	  comments += buf;
	}
	else if( in == "END_HEADER"){
	  state = Open;
	  return 1;
	}
	else {
	  cerr<<"Error: unknown field name.\n";
	  return 0;
	}
      }
      return 0;
    }
  }
}


int 
MPRead::getScalarVars(Array1< clString >& sv)
{
  int i;
  sv.remove_all();
  for( i = 0 ; i < (int)sVars.size(); i++)
    sv.add( sVars[i] );
  return 1;
}
 
int 
MPRead::getVectorVars(Array1< clString >& vv)
{
  int i;
  vv.remove_all();
  for( i = 0 ; i < (int)vVars.size(); i++)
    vv.add( vVars[i] );
  return 1;
}



int 
MPRead::ReadBlock( clString& datatype ) // either Grid or Particles 
{
  streampos mark = is.tellg();
  if( state != Open ){
    cerr<<"Error: file "<< filename <<" is not open for reading.\n";
    return 0;
  } else {
    clString in;
    datatype = "";
    while( is>>in){
      if(in == "MPD"){
        datatype += "HEADER";
        is.seekg(mark);
        return 1;
      } else if (in == "GRID"){
	datatype += "GRID";
	state = Grid;
	gridState = Info;
	return 1;
      } else if (in == "PARTICLES"){
	datatype += "PARTICLES";
	state = Particles;
	return 1;
      } else {
	cerr<<"Error: Unknown Block "<<in<<".\n";
	return 0;
      }
    }
    return 0;
  }
}


int 
MPRead::GetGridInfo( clString& name,
		     clString& type,
		     int& x, int& y, int& z,
		     Array1<clString>& sV,
		     Array1<clString>& vV)
{
  if( state != Grid){
    cerr<<"Error: not reading a grid\n";
    return 0;
  }
  if(gridState != Info){
    cerr<<"Error: you have already read the GridInfo\n";
    return 0;
  }

  sV.remove_all();
  vV.remove_all();
  sVars.remove_all();
  vVars.remove_all();
  
  const int buflen = 1024;
  char buf[buflen];
  clString bufout;

  is.getline(buf, buflen);
  istringstream in( buf );

  in >> name >> type >> x >> y >> z;

  SetCurrentType( type );
  x_size = x; y_size = y; z_size = z;

  is.getline(buf, buflen);
  istringstream sv( buf );
  while ( sv >> bufout ) {
    sVars.add( bufout );
    sV.add( bufout );
  }

  is.getline(buf, buflen);
  istringstream vv( buf );
  while ( vv >> bufout ) {
    vVars.add( bufout );
    vV.add( bufout );
  }
  
  gridState = Points;
  return 1;
}



int
MPRead::GetGridPoints( double& o_x, double& o_y, double& o_z,
		       double& dx, double& dy, double& dz )
{
  if( state != Grid ){
    cerr<<"Error: not reading a grid\n";
    return 0;
  } else if ( gridState != Points ){
    cerr<<"Error: you must first read the grid info\n";
    return 0;
  } else if (!(currentType == NC || currentType == CC || currentType == FC )){
    cerr<<"Error: wrong grid type for this function. \n"<<
      "Try : GetDataPoints(Array1<double>& X, Array1<double>& Y, " <<
      "Array1<double>& Z);\n";
    return 0;
  }
  const int buflen = 1024;
  char buf[buflen];

  is.getline(buf, buflen);
  istringstream in( buf );

  in >> o_x >> o_y >> o_z >> dx >> dy >> dz;
  
  minPt = Point(o_x, o_y, o_z);
  if(currentType == NC ){
    maxPt = Point(o_x + dx*(x_size-1),
		  o_y + dy*(y_size-1),
		  o_z + dz*(z_size-1));
  } else {
    maxPt = Point(o_x + dx*(x_size),
		  o_y + dy*(y_size),
		  o_z + dz*(z_size));
  }
  gridState = Scalars;
  return 1;
}
 
int
MPRead::GetGridPoints( Array1<double>& X,
		       Array1<double>& Y,
		       Array1<double>& Z)
{
   if( state != Grid ){
    cerr<<"Error: not reading a grid\n";
    return 0;
  } else if ( gridState != Points ){
    cerr<<"Error: you must first read the grid info\n";
    return 0;
  } else if (!(currentType == NC_i || currentType == CC_i 
	       || currentType == FC_i)){
    cerr<<"Error: wrong grid type for this function. \n"<<
      "Try : GetDataPoints(double& o_x, double& o_y, double& o_z, \n" <<
      "                    double& dx, double& dy, double& dz);\n";
    return 0;
  }

  X.remove_all(); X.resize(x_size);
  Y.remove_all(); Y.resize(y_size);
  Z.remove_all(); Z.resize(z_size);
  
  if(fileType == BIN){
    is.read((char *) &(X[0]), sizeof(double)*x_size);
    is.read((char *) &(Y[0]), sizeof(double)*y_size);
    is.read((char *) &(Z[0]), sizeof(double)*z_size);
  } else {
    double val;
    for(int i = 0; i < x_size; i++){ is >> val; X.add(val); }
    for(int i = 0; i < y_size; i++){ is >> val; Y.add(val); }
    for(int i = 0; i < z_size; i++){ is >> val; Z.add(val); }
  }
  minPt = Point(X[0],Y[0],Z[0]);
  maxPt = Point(X[X.size()-1], Y[Y.size()-1], Z[Z.size()-1]);
  
  gridState = Scalars;
  return 1;
}

// You can use dynamic type checking to check for the ScalarField type.
int
MPRead::GetScalarField( ScalarFieldHandle& sf )
{
  if(state != Grid){
    cerr<<"Error: not reading a grid.\n";
    return 0;
  } else if( svCount >= (int)sVars.size() ){
    cerr<<"Error: no more scalar values can be read.\n";
    return 0;
  } else if( gridState != Scalars ){
    cerr<<"Error:  you must read the data points first.\n";
    return 0;
  } else if( currentType == NC || currentType == CC){
    ScalarFieldRG *sfrg;
    if (currentType == NC) sfrg = scinew ScalarFieldRG();
    else  sfrg = scinew ScalarFieldRGCC();
    svCount++;
    cerr<<"Getting scalarfield of size "<< x_size <<" " << y_size <<
      " " << z_size <<endl;
    sfrg->resize(x_size, y_size, z_size);
    sfrg->set_bounds(minPt, maxPt);
    if(fileType == BIN ){
      is.read((char *)&(sfrg->grid(0,0,0)),
	      sizeof(double)*x_size*y_size*z_size);
    } else {
      for(int i = 0; i < x_size; i++){
	for(int j = 0; j < y_size; j++){
	  for(int k = 0; k < z_size; k++){
	    double val;
	    is >> val;
	    sfrg->grid(i,j,k) = val;
	  }
	}
      }
    }
      sf = ScalarFieldHandle(sfrg);
  } else {
    NOT_FINISHED("MPRead::GetScalarField( ScalarFieldHandle& sf )");
    return 0;
  }
  if (svCount == (int)sVars.size())
    gridState = Vectors;

  return 1;
}

// Generic ScalarField Reader 
int 
MPRead::GetScalarField( double *sf, int& length )
{
  if(state != Grid){
    cerr<<"Error: not reading a grid.\n";
    return 0;
  } else if( svCount >= (int)sVars.size() ){
    cerr<<"Error: no more scalar values can be read.\n";
    return 0;
  } else if( gridState != Scalars ){
    cerr<<"Error:  you must read the data points first.\n";
    return 0;
  } else if( currentType == NC || currentType == CC ){
    svCount++;
    length = x_size*y_size*z_size;
    sf = scinew double[length];
    if(fileType == BIN){
      is.read((char *) sf, sizeof(double)*length);
    } else {
      for(int i = 0; i < length; i++){
	is >> sf[i];
      }
    }
	    
  } else {
    NOT_FINISHED("MPRead::GetScalarField( double *sf, int& length )");
    return 0;
  }    
  if (svCount == (int)sVars.size())
    gridState = Vectors;

  return 1;
}

int
MPRead::GetVectorField( VectorFieldHandle& vf )
{
  if(state != Grid){
    cerr<<"Error: not reading a grid.\n";
    return 0;
  } else if ( vvCount >= (int)vVars.size() ) {
    cerr<<"Error: no more vector fields.\n";
    return 0;
  } else if(gridState != Vectors){
    cerr<<"Error: you haven't read the Scalar variables\n";
    return 0;
  } else if( currentType == NC || currentType == CC){
    vvCount++;
    VectorFieldRG *vfrg;
    if (currentType == NC ) vfrg  = scinew VectorFieldRG();
    else vfrg = scinew VectorFieldRGCC();
    vfrg->resize(x_size, y_size, z_size);
    vfrg->set_bounds(minPt, maxPt);    if(fileType == BIN ){
      double size = vfrg->grid.dim1()*vfrg->grid.dim2()*vfrg->grid.dim3();
      is.read((char *) &(vfrg->grid(0,0,0)), size*sizeof(double)*3);
    } else {
      for(int i = 0; i < vfrg->grid.dim1(); i++ )
	  for(int j = 0; j < vfrg->grid.dim2(); j++)
	    for(int k = 0; k < vfrg->grid.dim3(); k++){
	      double x,y,z;
	      is >> x >> y >> z;
	      vfrg->grid(i,j,k) = Vector(x,y,z);
	    }
    }
    vf = VectorFieldHandle(vfrg);

  } else {
    NOT_FINISHED("MPRead::GetVectorField( VectorFieldHandle& vf )");
    return 0;
  }

  if( vvCount == (int)vVars.size()){
    vvCount = 0;
    svCount = 0;
    state = Open;
    gridState = Empty;
  }
   return 1;
}

int
MPRead::GetVectorField( Vector *vf, int& length)
{
  if(state != Grid){
    cerr<<"Error: not reading a grid.\n";
    return 0;
  } else if ( vvCount >= (int)vVars.size() ) {
    cerr<<"Error: no more vector fields.\n";
    return 0;
  } else if(gridState != Vectors){
    cerr<<"Error: you haven't read the Scalar variables\n";
    return 0;
  } else if( currentType != FC && currentType != FC_i ){
    vvCount++;
    int i;
    length = x_size*y_size*z_size;
    vf = scinew Vector[length];
    for( i = 0; i < length; i++){
      is.read((char *) &(vf[i]), sizeof(double)*3);
    }
  } else {
    NOT_FINISHED("MPRead::GetVectorField( VectorFieldHandle& vf )");
    return 0;
  }

  if( vvCount == (int)vVars.size()){
    vvCount = 0;
    svCount = 0;
    state = Open;
    gridState = Empty;
  }
    
  return 1;
}



int 
MPRead::GetParticleInfo( clString& name,
			 int& N, // number of particles
			 Array1<clString>& s,
			 Array1<clString>& v)
{
  
  
  if( state != Particles ){
    cerr<<"Error: Not reading particles.\n";
    return 0;
  } else {

  s.remove_all();
  v.remove_all();
  psVars.remove_all();
  pvVars.remove_all();

    const int buflen = 1024;
    char buf[buflen];
    clString bufout;
    is.getline(buf,buflen);
    istringstream in( buf );

    in >> name >> N;
    nParticles = N;


    is.getline(buf,buflen);
    istringstream sv( buf );
    while( sv >> bufout ){
      s.add(bufout);
      psVars.add( bufout );
    }
    is.getline(buf,buflen);
    istringstream vv( buf );
    while( vv >> bufout ) {
      v.add( bufout );
      pvVars.add( bufout ); 
    }
    return 1;
  }
}
  
  // this can only be called N times after

int 
MPRead::GetParticle( Point& p,
		     Array1< double >& scalars,
		     Array1< Vector >& vectors)
{
  int i;
  if( state != Particles ){
    cerr<<"Error: Not reading particles.\n";
    return 0;
  } else if(pCount >= nParticles) {
    cerr<<"Error: All particles have been read\n";
    return 0;
  } else {
    pCount++;
    scalars.remove_all();
    vectors.remove_all();
    if(fileType == BIN) {
      scalars.setsize( psVars.size() );
      vectors.setsize( pvVars.size() );
      is.read((char *) &p, sizeof(double)*3);
      is.read((char *) &(scalars[0]), sizeof(double)*psVars.size());
      for(i = 0; i < (int)pvVars.size(); i++){
	is.read((char *)&(vectors[i]), sizeof(double)*3);
      }
    } else {
      double x,y,z;
      is >> x >> y >> z;
      p = Point(x,y,z);
      for( i = 0; i < (int)psVars.size(); i++){
	is >> x;
	scalars.add(x);
      }
      for( i = 0; i < (int)pvVars.size(); i++) {
	is >> x >> y >> z;
	vectors.add(Vector(x,y,z));
      }
    }
    if( pCount == nParticles) {
      state = Open;
      nParticles = 0;
      pCount = 0;
    }
    return 1;
  }
}


int
MPRead::GetParticleVariableValue( int pid,
				  clString pSetName,
				  clString varname,
				  double& value)
{
  clString  in;
  clString name, sv, vv, filetype, comment;
  clString type;
  double tmp;
  double ox,oy,oz, dx,dy,dz;
  Array1<double> X,Y,Z;
  Array1<clString> s,v;
  int i, multiplyer, N;
  bool varfound = false;
  streampos mark  = is.tellg();
  while( ReadBlock( type )){
    if(type == "HEADER"){
      is.seekg(mark);
      ReadHeader(name, filetype, comment);
    } else if (type  == "GRID"){
      GetGridInfo( name, type, x_size, y_size, z_size, s, v);
      N = x_size*y_size*z_size;
      if( currentType == NC || currentType == CC || currentType == FC_i ){
	GetGridPoints( ox,oy,oz, dx,dy,dz);
	multiplyer = N*(sVars.size() + 3*vVars.size());
	if( currentType == NC || currentType == CC ){
	  is.seekg(sizeof(double)*multiplyer, ios::cur);
	}
      } else {
	GetGridPoints( X, Y, Z );
	is.seekg(sizeof(double)*x_size*y_size*z_size, ios::cur);
	if( currentType != FC_i ){
	  for(i = 0; i < (int)vVars.size(); i++)
	  NOT_FINISHED("MPRead::GetParticleVariableValue for staggered grids");
	    is.seekg(sizeof(double)*3*x_size*y_size*z_size, ios::cur);
	} else {
	  NOT_FINISHED("MPRead::GetParticleVariableValue for staggered grids");
	    is.seekg(sizeof(double)*3*x_size*y_size*z_size*3, ios::cur);
	}
      }
      state = Open;
      gridState = Empty;
    } else if (type == "PARTICLES"){
      GetParticleInfo( name, N, s, v);
      if( name == pSetName ){
	for( i = 0; i < (int)s.size(); i++){
	  cerr<< "Reading varname "<< s[i] << "looking for varname "<< varname <<endl;
	  if( s[i] == varname ){
	    varfound = true;
	    break;
	  }
	}
	if(varfound){
	  is.seekg(sizeof(double)*(3+s.size()+ v.size()*3)*pid, ios::cur);
	  is.seekg(sizeof(double)*i, ios::cur);
	  is.read((char *) &value, sizeof(double));
	  cerr<<"Value = "<< value << endl;
	  return 1;
	} else {
	  return 0;
	}
      } else {
	is.seekg(sizeof(double)*(3+s.size()+ v.size()*3)*N,ios::cur);
      }

      state = Open;
    } else {
      cerr<<"Error: Unknown Block "<<in<<".\n";
      return 0;
    }
    mark = is.tellg();
  }
  return 0;
}

void
MPRead::SetCurrentType( const clString&  type )
{
  if( type == "NC" ) currentType = NC;
  else if( type == "CC") currentType = CC;
  else if( type == "FC") currentType = FC;
  else if( type == "NC_i") currentType = NC_i;
  else if( type == "CC_i") currentType = CC_i;
  else if( type == "FC_i") currentType = FC_i;
  else cerr<<"Error: Unknown Grid type\n";
}

} // end namespace Modules
} // end namespace Uintah
