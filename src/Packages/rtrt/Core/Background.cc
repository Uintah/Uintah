#ifndef BACKGROUND_H
#include <Packages/rtrt/Core/Background.h>
#include <Core/Math/MiscMath.h>
#endif

#include <fstream>
#include <Core/Geometry/Vector.h>

using namespace rtrt;
using namespace SCIRun;

Background::Background(const Color& avg)
  : avg(avg), origAvg_( avg )
{
}

Background::~Background() {}


// *****************************************************************
//     ConstantBackground members

ConstantBackground::ConstantBackground(const Color& C) : 
  Background(C), C(C), origC_(C) {}

ConstantBackground::~ConstantBackground() {}

void ConstantBackground::color_in_direction( const Vector&, Color& result) const
{
  result=C;
}

// *****************************************************************
//     LinearBackground members


LinearBackground::~LinearBackground() {}

LinearBackground::LinearBackground( const Color& C1, 
				    const Color& C2,
				    const Vector& direction_to_C1) :
  Background(C1), C1(C1), C2(C2), origC1_(C1), origC2_(C2),
  direction_to_C1(direction_to_C1)
{
}

void LinearBackground::color_in_direction(const Vector& v, 
					  Color& result) const 
{
    double t = 0.5* (1 + Dot(v, direction_to_C1 ) );
    result=t*C1 + (1-t)*C2;
}

inline  int IndexOfMinAbsComponent( const Vector& v ) 
{
    if ( fabs( v[0]) < fabs( v[1]) && fabs(v[0]) < fabs(v[2]))
            return 0;
        else if( fabs(v[1]) < fabs(v[2] ) )
            return 1;
        else
            return 2;
    }

inline Vector PerpendicularVector( const Vector& v ) 
{
   int axis = IndexOfMinAbsComponent( v );
   if( axis == 0 )
      return Vector( 0.0, v[2], -v[1] );
   else if ( axis == 1 )
      return Vector( v[2], 0.0, -v[0] );
   else
      return Vector( v[1], -v[0], 0.0 );
   }


EnvironmentMapBackground::EnvironmentMapBackground( char* filename,
						    const Vector& up ) :
    Background( Color( 0, 0, 0 ) ),
    _width( 0 ),
    _height( 0 ),
    _aspectRatio( 1.0 ),
    _text( 0 ),
    _up( up ),
    ambientScale_( 1.0 )
{
  //
  // Built an orthonormal basis
  //
  _up.normalize();
  _u = PerpendicularVector( _up );
  _v = Cross( _up, _u );
  read_image( filename );
  
  cout << "env_map width, height: " << _width << ", " << _height << endl;
}

EnvironmentMapBackground::~EnvironmentMapBackground( void )
{
    if( _text ) {
	if( _text->texImage )
	    free( _text->texImage );
	if( _text->texImagef )
	    free( _text->texImagef );
	free( _text );
    }
}

void
EnvironmentMapBackground::color_in_direction( const Vector& DIR , Color& result) const
{
    //
    // Convert to local basis
    //
    Vector dir = ChangeToBasis( DIR );

    // cerr << "direction  = " << dir << endl;
    // Map direction vector dir to (u,v) coordinates
    // cerr << "Length = " << dir.length() << endl;
    //double r = atan2( sqrt( dir.x()*dir.x() + dir.y()*dir.y() ), dir.z() );
    //r /= M_PI;  /* -0.5 .. 0.5 */
    //double phi = atan2( dir.y(), dir.x() );
    double r =  sqrt( dir.x()*dir.x() + dir.y()*dir.y() );
    double v = atan2( dir.x(), dir.y() ) / ( 2.0*M_PI ) + 0.5;
    double u = atan2( r, dir.z() ) / M_PI;
    // double u = ( ( atan2( dir.y(), dir.x() ) + M_PI ) / ( 2.0*M_PI ) );
    // double v = ( ( asin( dir.z() ) + (0.5*M_PI) ) / M_PI );
    //double u = Clamp( r * cos( phi ) + 0.5, 0.0, 1.0 );
    //double v = Clamp( r * sin( phi ) + 0.5, 0.0, 1.0 );

    //double l1 = sqrt(dir.x()*dir.x()+dir.y()*dir.y());
    //double l2 = sqrt(dir.x()*dir.x()+dir.y()*dir.y()+dir.z()*dir.z());
    //double v = asin(dir.z()/l2);
    //double u = (dir.x()<0)?M_PI-asin(dir.y()/l1):asin(dir.y()/l1);      

    //dir.normalize();
    
    //double v = (DIR.x()+1)/2.;
    //double u = (DIR.z()+1)/2.;

    result = _image( int( v*( _width - 1 ) ), int( u*( _height - 1 ) ) ) *
      ambientScale_;
}

static void eat_comments_and_whitespace(ifstream &str)
{
  char c;
  str.get(c);
  for(;;) {
    if (c==' '||c=='\t'||c=='\n') {
      str.get(c);
      continue;
    } else if (c=='#') {
      str.get(c);
      while(c!='\n')
        str.get(c);
    } else {
      str.unget();
      break;
    }
  }
}

void 
EnvironmentMapBackground::read_image( char* filename ) 
{
#if 0
  _text = ReadPPMTexture( filename );
  
  char* color = _text->texImage;
  _width = _text->img_width;
  _height = _text->img_height;
  _image.resize( _width, _height );
  _aspectRatio = _height / _width;

  for( int i = 0; i < _width; i++ ) {
    for( int j = 0; j < _height; j++ ) {
      double r = color[0] / 255.;
      double g = color[1] / 255.;
      double b = color[2] / 255.;
      color += 4;
      _image( i, j ) = Color( r, g, b );
    }
  }

  if( _text ) {
      if( _text->texImage )
	  free( _text->texImage );
      if( _text->texImagef )
	  free( _text->texImagef );
      free( _text );
  }

#else

  //unsigned nu, nv;
  double size;
  ifstream indata(filename);
  unsigned char color[3];
  string token;

  if (!indata.is_open()) {
    cerr << "ImageMaterial: WARNING: I/O fault: no such file: " << filename << endl;
  }
    
  indata >> token; // P6
  eat_comments_and_whitespace(indata);
  indata >> _width >> _height;
  eat_comments_and_whitespace(indata);
  indata >> size;
  eat_comments_and_whitespace(indata);
  _image.resize(_width, _height);
  for(unsigned v=0;v<_height;++v){
    for(unsigned u=0;u<_width;++u){
      indata.read((char*)color, 3);
      double r=color[0]/size;
      double g=color[1]/size;
      double b=color[2]/size;
      _image(u,v)=Color(r,g,b);
    }
  }

  //valid_ = true;
#endif
}

