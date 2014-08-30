#ifndef Test_Functions_h
#define Test_Functions_h

#include <spatialops/FieldFunctions.h>

//====================================================================


template<typename FieldT>
class LinearFunction : public SpatialOps::FieldFunction3D<FieldT>
{
public:
  LinearFunction( const FieldT& x, const FieldT& y, const FieldT& z )
    : SpatialOps::FieldFunction3D<FieldT>(x,y,z)
  {}
  ~LinearFunction(){}
  void evaluate( FieldT& phi ) const
  {
    const FieldT& x=this->get_x();
    const FieldT& y=this->get_y();
    const FieldT& z=this->get_z();
    phi <<= 2*x+3*y+4*z;
  }

  void dx( FieldT& gradPhi ) const{ gradPhi = 2.0; }
  void dy( FieldT& gradPhi ) const{ gradPhi = 3.0; }
  void dz( FieldT& gradPhi ) const{ gradPhi = 4.0; }

  void d2x( FieldT& d2phi ) const{ d2phi=0.0; }
  void d2y( FieldT& d2phi ) const{ d2phi=0.0; }
  void d2z( FieldT& d2phi ) const{ d2phi=0.0; }

private:
};


//====================================================================


template<typename FieldT>
class SinFun : public SpatialOps::FieldFunction3D<FieldT>
{
public:

  SinFun( const FieldT& x, const FieldT& y, const FieldT& z )
    : SpatialOps::FieldFunction3D<FieldT>(x,y,z),
      pi( 3.141592653589793 )
  {}
  ~SinFun(){}
  void evaluate( FieldT& phi ) const
  {
    const FieldT& x=this->get_x();
    const FieldT& y=this->get_y();
    const FieldT& z=this->get_z();
    phi <<= sin(x*pi) + sin(0.5*pi*y) + cos(pi*z) + 1.0;
  }

  void dx( FieldT& gradPhi ) const
  {
    const FieldT& x=this->get_x();
    gradPhi <<= pi*cos(x*pi);
  }

  void dy( FieldT& grad ) const
  {
    const FieldT& y=this->get_y();
    grad <<= 0.5*pi*cos(0.5*pi*y);
  }

  void dz( FieldT& grad ) const
  {
    const FieldT& z=this->get_z();
    grad <<=  -pi*sin(pi*z);
  }


  void d2x( FieldT& d2phi ) const
  {
    const FieldT& x=this->get_x();
    d2phi <<= -pi*pi*sin(x*pi);
  }


  void d2y( FieldT& d2phi ) const
  {
    const FieldT& y=this->get_y();
    d2phi <<= -0.25*pi*pi*sin(0.5*pi*y);
  }

  void d2z( FieldT& d2phi ) const
  {
    const FieldT& z=this->get_z();
    d2phi <<= -pi*pi*cos(z*pi);
  }

private:
  const double pi;
};


#endif // Test_Functions_h
