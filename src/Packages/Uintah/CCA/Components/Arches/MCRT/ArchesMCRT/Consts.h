// define a couple of consts
// gray, diffuse surfaces, diffusely emission and absorption
// uniform temperature on each subsurface



#include <cmath>

const bool isotropic = true; // isotropic scattering, phase function = 1

static const int TOP = 0;
static const int BOTTOM = 1;
static const int FRONT = 2;
static const int BACK = 3;
static const int LEFT = 4;
static const int RIGHT = 5;
static const double pi = 4 * atan(1);
static const double SB = 5.669 * pow(10., -8);

// all normal vectors pointing inward the volume
static const double n_top[3] = { 0, 0, -1};
static const double n_bottom[3] = {0, 0, 1};
static const double n_front[3] = {0, 1, 0}; // so y direction from front to back
static const double n_back[3] = {0, -1, 0};
static const double n_left[3] = {1, 0, 0};
static const double n_right[3] = {-1, 0, 0};


// has to be static, not clear why?
static const double *surface_n[6] = { n_top, n_bottom,
				      n_front, n_back,
				      n_left, n_right};






