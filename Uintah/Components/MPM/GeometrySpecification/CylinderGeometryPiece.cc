#include "CylinderGeometryPiece.h"

CylinderGeometryPiece::CylinderGeometryPiece() {}

CylinderGeometryPiece::CylinderGeometryPiece(AXIS a, Point o,
					     double l, double r) :
  d_axis(a),d_origin(o),d_length(l),d_radius(r)
{
}


CylinderGeometryPiece::~CylinderGeometryPiece()
{
}

int CylinderGeometryPiece::checkShapesPositive(Point check_point, int &np,
					       int piece_num,
					       Vector particle_spacing,
					       int ppold)
{   
  int pp;

  // Positive
  // gb[1] = axis;
  // gb[2-4] = origin;
  // gb[5] = length;
  // gb[6] = radius;

  

  double dlpp=particle_spacing.x();
  if(dlpp<particle_spacing.y()){
    dlpp=particle_spacing.y();
  }
  if(dlpp<particle_spacing.z()){
    dlpp=particle_spacing.z();
  }

  double rsqr = d_radius * d_radius;
  double rdpsqr= (d_radius+dlpp) * (d_radius+dlpp);

  Point tmp_point = check_point - d_origin;
  Vector test_point(tmp_point.x(),tmp_point.y(),tmp_point.z());
  double len;
  // Check for each d_axis aligned case
  switch(d_axis) {
  case X :   // X axis
    len = test_point.length() - (test_point.x() * test_point.x());
    if( (len*len <= rsqr) && 
	(test_point.x() - particle_spacing.x() <=0.0)
       && (d_origin.x()+ d_length +
	  particle_spacing.x()-check_point.x()>=0.0)){
      pp = 2;
      inPiece=piece_num;
      if((test_point.x() <= particle_spacing.x()) &&
	 (test_point.x() < 0.0)){
	pp = -7;
      }
      else if((d_origin.x()+d_length-check_point.x()>=
	       -particle_spacing.x())&&
	      (d_origin.x()+d_length-check_point.x() < 0.0) ){
	pp = -8;
      }
    }
    else if((pow((check_point.y()-d_origin.y()),2.0)+
	     pow((check_point.z()-d_origin.z()),2.0)>rsqr)
	    &&(pow((check_point.y()-d_origin.y()),2.0)+
	       pow((check_point.z()-d_origin.z()),2.0)<=rdpsqr)
	    &&(check_point.x()>=d_origin.x())&&
	    (check_point.x()<(d_origin.x()+d_length)))
      {
	pp = -9;
      }
    break;
  case Y : // Y axis
    if((pow((check_point.x()-d_origin.x()),2.0)+
	pow((check_point.z()-d_origin.z()),2.0) <= rsqr)
       &&(d_origin.y()-particle_spacing.y()-check_point.y() <= 0.0)
       &&(d_origin.y()+d_length+particle_spacing.y()-
	  check_point.y()>=0.0)){
      pp = 2;
      inPiece=piece_num;
      if((d_origin.y()-check_point.y() <= particle_spacing.y())
	 &&(d_origin.y()-check_point.y()>0.0)){
	pp = -7;
      }
      else if((d_origin.y()+d_length-check_point.y()>=
	       -particle_spacing.y())&&
	      (d_origin.y()+d_length-check_point.y() < 0.0) ){
	pp = -8;
      }
    }
    else if((pow((check_point.x()-d_origin.x()),2.0)+
	     pow((check_point.z()-d_origin.z()),2.0)>rsqr)
	    &&(pow((check_point.x()-d_origin.x()),2.0)+
	       pow((check_point.z()-d_origin.z()),2.0)<=rdpsqr)
	    &&(check_point.y()>=d_origin.y())
	    &&(check_point.y()<(d_origin.y()+d_length)))
      {
	pp = -9;
      }
    break;
  case Z : // Z axis
    if((pow((check_point.x()-d_origin.x()),2.0)+
	pow((check_point.y()-d_origin.y()),2.0) <= rsqr)
       &&(d_origin.z()-particle_spacing.z()-check_point.z() <= 0.0)
       &&(d_origin.z()+d_length+particle_spacing.z()-
	  check_point.z()>=0.0)){
      pp = 2;
      inPiece=piece_num;
      if((d_origin.z()-check_point.z() <= particle_spacing.z())
	 &&(d_origin.z()-check_point.z()>0.0)){
	pp = -7;
      }
      else if((d_origin.z()+d_length-check_point.z()>=
	       -particle_spacing.z())&&
	      (d_origin.z()+d_length-check_point.z() < 0.0) ){
	pp = -8;
      }
    }
    else if((pow((check_point.x()-d_origin.x()),2.0)+
	     pow((check_point.y()-d_origin.y()),2.0)>rsqr)
	    &&(pow((check_point.x()-d_origin.x()),2.0)+
	       pow((check_point.y()-d_origin.y()),2.0)<=rdpsqr)
	    &&(check_point.z()>=d_origin.z())&&(check_point.z()
						<(d_origin.z()+d_length)))
      {
	pp = -9;
      }
    break;
  } // End of switch for PosNeg > 0
  
  
  return pp;
}

int CylinderGeometryPiece::checkShapesNegative(Point check_point, int &np,
					       int piece_num, 
					       Vector particle_spacing,
					       int ppold)
{   
  int pp;

  // Positive
  // gb[1] = axis;
  // gb[2-4] = origin;
  // gb[5] = length;
  // gb[6] = radius;

  double dlpp=particle_spacing.x();
  if(dlpp<particle_spacing.y()){
    dlpp=particle_spacing.y();
  }
  if(dlpp<particle_spacing.z()){
    dlpp=particle_spacing.z();
  }

  d_radius = d_radius;
  double rsqr = d_radius * d_radius;
  double rdpsqr=pow(d_radius+dlpp,2.0);

    // Negative 
    switch (d_axis) {
    case X :
	if(pow((check_point.y()-d_origin.y()),2.0)+pow((check_point.z()-d_origin.z()),2.0) < pow(d_radius,2.0)) {
	  pp = -91;
	  np = piece_num;
	  if((check_point.x()<d_origin.x())||(check_point.x()>(d_origin.x()+d_length))){
	    pp = ppold;
	  }
	  else if((check_point.x()-d_origin.x() <= particle_spacing.x()) &&
		  (check_point.x()-d_origin.x() > 0.0 )  &&
		  (pp != -91)){
	    pp = -71;
	  }
	  else if((check_point.x()-(d_origin.x()+d_length) >= -particle_spacing.x()) &&
		  (check_point.x()-(d_origin.x()+d_length) < 0.0 )   &&
		  (pp != -91)){
	    pp = -81;
	  }
	}
      break;
    case Y : 
	if(pow((check_point.x()-d_origin.x()),2.0)+pow((check_point.z()-d_origin.z()),2.0) < pow(d_radius,2.0)) {
	  pp = -91;
	  np = piece_num;
	  if((check_point.y()<d_origin.y())||(check_point.y()>(d_origin.y()+d_length))){
	    pp = ppold;
	  }
	  else if((check_point.y()-d_origin.y() <= particle_spacing.y()) &&
		  (check_point.y()-d_origin.y() > 0.0 )  &&
		  (pp != -91)){
	    pp = -71;
	  }
	  else if((check_point.y()-(d_origin.y()+d_length) >= -particle_spacing.y()) &&
		  (check_point.y()-(d_origin.y()+d_length) < 0.0 )  &&
		  (pp != -91)){
	    pp = -81;
	  }
	}
      break;
    case Z :
	if(pow((check_point.x()-d_origin.x()),2.0)+pow((check_point.y()-d_origin.y()),2.0) < pow(d_radius,2.0)) {
	  pp = -91;
	  np = piece_num;
	  if((check_point.z()<d_origin.z())||(check_point.z()>(d_origin.z()+d_length))){
	    pp = ppold;
	  }
	  else if((check_point.z()-d_origin.z() <= particle_spacing.z()) &&
		  (check_point.z()-d_origin.z() > 0.0 )  &&
		  (pp != -91)){
	    pp = -71;
	  }
	  else if((check_point.z()-(d_origin.z()+d_length) >= -particle_spacing.z()) &&
		  (check_point.z()-(d_origin.z()+d_length) < 0.0 )  &&
		  (pp != -91)){
	    pp = -81;
	  }
	}
      break;
    }
  
  
  return pp;
}


void CylinderGeometryPiece::computeNorm(Vector &norm, Point part_pos, 
						int sf[7], int inPiece, 
						int &np)
{

  Vector dir(0.0,0.0,0.0);
  norm.set(0.0,0.0,0.0);
  int small = 1;

  for(int i=1;i<=6;i++) {
    if(sf[i] < small) { 
      small = sf[i]; 
    }
  }
  
  // Not a surface point
  if(small == 1){ 
    return; 
  }		

  switch (d_axis) {
  case X:
    // x-axis aligned
    for(int i=1;i<=6;i++){
      if(sf[i]==(-7)){	
	// low x
	dir.set(-1.0,0.0,0.0);
	norm+=dir;
      }
      if(sf[i]==(-8)){		
	// high x
	dir.set(1.0,0.0,0.0);
	norm+=dir;
      }
      if(sf[i]==(-9)){		
	// curved surface
	Point tmp = (part_pos - d_origin)*2.0;
	dir.set(0.0,tmp.y(),tmp.z());
	dir.normalize();
	norm+=dir;
      }
    }
    break;
  case Y: 
    // y-axis aligned
    for(int i=1;i<=6;i++){
      if(sf[i]==(-7)){		
	// low y
	dir.set(0.0,-1.0,0.0);
	norm+=dir;
      }
      if(sf[i]==(-8)){		
	// high y
	dir.set(0.0,1.0,0.0);
	norm+=dir;
      }
      if(sf[i]==(-9)){		
	// curved surface
	Point tmp = (part_pos - d_origin)*2.0;
	dir.set(tmp.x(),0.0,tmp.z());
	dir.normalize();
	norm+=dir;
      }
    }
    break;
  case Z: 
    // z-axis aligned
    for(int i=1;i<=6;i++){
      if(sf[i]==(-7)){		
	// low z
	dir.set(0.0,0.0,-1.0);
	norm+=dir;
      }
      if(sf[i]==(-8)){		
	// high z
	dir.set(0.0,0.0,1.0);
	norm+=dir;
      }
      if(sf[i]==(-9)){		
	// curved surface
	Point tmp = (part_pos - d_origin)*2.0;
	dir.set(tmp.x(),tmp.y(),0.0);
	dir.normalize();
	norm+=dir;
      }
    }
    break;
  }
    // The point is on the surface of a "negative" object.
    // Get the geometry information for the negative object so we can determine
    // a surface normal.
    if(small < -10){	
   
      switch (d_axis) {
      case X:
   	// x-axis aligned
	for(int i=1;i<=6;i++){
          if(sf[i]==(-71)){	
	    // low x
	    dir.set(1.0,0.0,0.0);
	    norm+=dir;
          }
          if(sf[i]==(-81)){		
	    // high x
	    dir.set(-1.0,0.0,0.0);
	    norm+=dir;
          }
          if(sf[i]==(-91)){
	    // curved surface
	    Point tmp = (part_pos - d_origin)*(-2.0);
	    dir.set(0.0,tmp.y(),tmp.z());
	    dir.normalize();
	    norm+=dir;
          }
	}
      case Y:
      // y-axis aligned
      for(int i=1;i<=6;i++){
	if(sf[i]==(-71)){	
	  // low y
	  dir.set(0.0,1.0,0.0);
	  norm+=dir;
	}
	if(sf[i]==(-81)){		
	  // high y
	  dir.set(0.0,-1.0,0.0);
	  norm+=dir;
	}
	if(sf[i]==(-91)){		
	  // curved surface
	  Point tmp = (part_pos - d_origin)*(-2.0);
	  dir.set(tmp.x(),0.0,tmp.z());
	  dir.normalize();
	  norm+=dir;
	}
      }
    case Z:
      // z-axis aligned
      for(int i=1;i<=6;i++){
	if(sf[i]==(-71)){	
	  // low z
	  dir.set(0.0,0.0,1.0);
	  norm+=dir;
	}
	if(sf[i]==(-81)){	
	  // high z
	  dir.set(0.0,0.0,-1.0);
	  norm+=dir;
	}
	if(sf[i]==(-91)){		
	  // curved surface
	  Point tmp = (part_pos - d_origin)*(-2.0);
	  dir.set(tmp.x(),tmp.y(),tmp.z());
	  dir.normalize();
	  norm+=dir;
	}
      }
      } 
    }

    norm.normalize();

}

// $Log$
// Revision 1.1  2000/04/14 02:05:45  jas
// Subclassed out the GeometryPiece into 4 types: Box,Cylinder,Sphere, and
// Tri.  This made the GeometryObject class simpler since many of the
// methods are now relegated to the GeometryPiece subclasses.
//
