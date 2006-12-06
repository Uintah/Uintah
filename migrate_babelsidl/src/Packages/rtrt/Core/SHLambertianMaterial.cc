//========================================================================
//
// Filename: SHLambertianMaterial.cc
//
//
// Material used for rendering irradiance environment maps with 
// spherical harmonic coefficients.
//
//
//
//
// Reference: This is an implementation of the method described by
//            Ravi Ramamoorthi and Pat Hanrahan in their SIGGRAPH 2001 
//            paper, "An Efficient Representation for Irradiance
//            Environment Maps".
//
//========================================================================

#include <Packages/rtrt/Core/SHLambertianMaterial.h>

#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/ppm.h>
#include <Packages/rtrt/Core/prefilter.h>

#include <math.h>

using namespace rtrt;
using namespace std;

const float c1 = 0.429043;
const float c2 = 0.511664;
const float c3 = 0.743125;
const float c4 = 0.886227;
const float c5 = 0.247708;

SHLambertianMaterial::SHLambertianMaterial( const Color& R, char* envmap, 
					    float scale, int type ) :
    albedo( R ), 
    fudgeFactor( scale )
{
    SHCoeffs c;
    texture* tex = 0;
    if( type == 0 ) 
	tex = ReadPPMTexture( envmap );
    else if( type == 1 )
	tex = ReadRGBETexture( envmap );
    else if( type == 2 ) {
	cerr << "PREDEFINED COEFFS" << endl;
	// 
	// Default coefficients -- mainly for debugging
	//
	// These are some default coefficients 
	// The coefficients come from the Paul Debevec's Grace Cathedral 
	// light probe image
	//
	L00  = Color( 0.078908,  0.043710,  0.054161 );
	L1_1 = Color( 0.039499,  0.034989,  0.060488 );
	L10  = Color(-0.033974, -0.018236, -0.026940 );
	L11  = Color(-0.029213, -0.005562,  0.000944 );
	L2_2 = Color(-0.011141, -0.005090, -0.012231 );
	L2_1 = Color(-0.026240, -0.022401, -0.047479 );
	L20  = Color(-0.015570, -0.009471, -0.014733 );
	L21  = Color( 0.056014,  0.021444,  0.013915 );
	L22  = Color( 0.021205, -0.005432, -0.030374 );
    }
    
    if( tex && type != 2 ) {

	prefilter( tex, c );
	
	L00 = Color( c.coeffs[0][0], c.coeffs[0][1], c.coeffs[0][2] );
	L1_1 = Color( c.coeffs[1][0], c.coeffs[1][1], c.coeffs[1][2] );
	L10 = Color( c.coeffs[2][0], c.coeffs[2][1], c.coeffs[2][2] );
	L11 = Color( c.coeffs[3][0], c.coeffs[3][1], c.coeffs[3][2] );
	L2_2 = Color( c.coeffs[4][0], c.coeffs[4][1], c.coeffs[4][2] );
	L2_1 = Color( c.coeffs[5][0], c.coeffs[5][1], c.coeffs[5][2] );
	L20 = Color( c.coeffs[6][0], c.coeffs[6][1], c.coeffs[6][2] );
	L21 = Color( c.coeffs[7][0], c.coeffs[7][1], c.coeffs[7][2] );
	L22 = Color( c.coeffs[8][0], c.coeffs[8][1], c.coeffs[8][2] );
    
    } 

    if( tex ) {
	if( tex->texImage )
	    free( tex->texImage );
	if( tex->texImagef )
	    free( tex->texImagef );
	free( tex );
    }

}

SHLambertianMaterial::~SHLambertianMaterial( void )
{
}

void 
SHLambertianMaterial::shade( Color& result, const Ray& ray,
			     const HitInfo& hit, int depth,
			     double , const Color& ,
			     Context* cx )
{
    double nearest=hit.min_t;
    Object* obj=hit.hit_obj;
    Point hitpos(ray.origin()+ray.direction()*nearest);
    Vector normal(obj->normal(hitpos, hit));
    double incident_angle=-Dot(normal, ray.direction());
    if(incident_angle<0){
	incident_angle=-incident_angle;
	normal=-normal;
    }
    
    // result = albedo * ( irradCoeffs( normal ) + ambient(cx->scene, hitpos, normal ) );
    result = fudgeFactor * albedo * irradCoeffs( normal );
    int ngloblights=cx->scene->nlights();
    int nloclights=my_lights.size();
    int nlights=ngloblights+nloclights;
    cx->stats->ds[depth].nshadow+=nlights;
    for(int i=0;i<nlights;i++){
        Light* light;
        if (i<ngloblights)
	  light=cx->scene->light(i);
	else 
	  light=my_lights[i-ngloblights];

	if( !light->isOn() )
	  continue;

	Vector light_dir=light->get_pos()-hitpos;
	double dist=light_dir.normalize();
	Color shadowfactor(1,1,1);
	if(cx->scene->lit(hitpos, light, light_dir, dist, shadowfactor, depth, cx) ){
	    double cos_theta=Dot(light_dir, normal);
	    if(cos_theta < 0){
		cos_theta=-cos_theta;
		light_dir=-light_dir;
	    }
	    //
	    // Note that we are multiplying by the spherical harmonics coefficients
	    //
	    result+=light->get_color()*albedo*(cos_theta*shadowfactor);
	} else {
	    cx->stats->ds[depth].inshadow++;
	}
    }
}

//
// This is implementation of equation 13 from Ramamoorthi and Hanrahan 
//

Color 
SHLambertianMaterial::irradCoeffs( const Vector& N ) const
{
    //------------------------------------------------------------------
    // These are variables to hold x,y,z and squares and products
    
    //
    // Could be made more efficient by using vector N directly without 
    // assignment
    //
    float x = N.x();
    float y = N.y();
    float z = N.z();
    float x2 = x*x;
    float y2 = y*y;
    float z2 = z*z;
    float xy = x*y;
    float yz = y*z;
    float xz = x*z;

    //------------------------------------------------------------------ 
    // Finally, we compute equation 13
    
    Color col = c1*L22*(x2-y2) + c3*L20*z2 + c4*L00 - c5*L20 
	+ 2*c1*(L2_2*xy + L21*xz + L2_1*yz) 
	+ 2*c2*(L11*x+L1_1*y+L10*z);

    // cerr << "Color = " << col << endl;
    return col;

}


