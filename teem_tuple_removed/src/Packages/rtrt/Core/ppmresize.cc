#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <Packages/rtrt/Core/Color.h>
#include <mtl/matrix.h>
#include <mtl/mtl.h>

using namespace mtl;
using namespace std;
using namespace rtrt;

typedef matrix<float>::type L_Matrix;
typedef dense1D<Color> ColorV;

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
// bspline code

class Knot_01 {
private:
  int _num_pts;
  int _degree;
  float inc;
public:
  Knot_01(const int num_pts, const int degree):
    _num_pts(num_pts), _degree(degree), inc(1/(float)(num_pts - degree)) {}

  float operator[](int n) const {
    //    ASSERTRANGE(n, 0, _num_pts + _degree + 1);
    if (n < 0 || n >= (_num_pts + _degree + 1)) {
      cerr << "Bad Value given to Knot_01" << n << endl;
      exit(1);
    }
    if (n > _degree) {
      if (n < _num_pts) {
	// do the middle case
	return (n - _degree) * inc;
      } else {
	// do the end case
	return 1;
      }
    } else {
      // do the beginning case
      return 0;
    }
  }

  inline int size() const { return (_num_pts + _degree + 1); }
  inline int degree() const { return _degree; }
  inline float domain_min() const { return 0; }
  inline float domain_max() const { return 1; }

  int find_next_index(const float t) const {
    int t_trunk = (int) (t / inc) + _degree;
    if ((*this)[t_trunk] <= t && t < (*this)[t_trunk+1]) {
      return t_trunk;
    } else {
      for (int i = _degree; i < _num_pts; i++) {
	if ((*this)[i] <= t && t < (*this)[i+1]) {
	  //    last_J = i;
	  return i;
	}
      }
    }
    // should never reach this point
    return _degree;
  }
};

float basis(const int i, const int k, const Knot_01& knotV,const float t) {
  if (k > 0) {
    float result;
    if (knotV[i] < knotV[i+k]) {
      result = (t - knotV[i]) / (knotV[i+k] - knotV[i]) *
	basis(i, k-1, knotV, t);
    } else {
      result = 0;
    }
    if (knotV[i+1] < knotV[i+k+1]) {
      result += (knotV[i+k+1] - t) / (knotV[i+k+1] - knotV[i+1]) *
	basis(i+1, k-1, knotV, t);
    }
    return result;
  } else {
    if (knotV[i] <= t && t < knotV[i+1])
      return 1;
    else
      return 0;
  }
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
// ppm code

void write_ppm(Color** buffer, const int width, const int height,
	       const char * filename) {

  // append the correct file extension to the filename.
  printf("opening file for writing: %s\n",filename);
  FILE *out = fopen(filename, "w");
  if (!out) {
    fprintf(stderr, "Can't open output file %s\n", filename);
    return;
  }

  fprintf(out, "P6 ");   // color rawbits format
  fprintf(out, "%d %d %d ", width, height, 255);  // width, height, and depth
  
  for (int y = 0; y < width; y++) {
    for (int x = 0; x < height; x++) {
      unsigned char r = (unsigned char)(buffer[x][y].red() * 255);
      unsigned char g = (unsigned char)(buffer[x][y].green() * 255);
      unsigned char b = (unsigned char)(buffer[x][y].blue() * 255);
      fprintf(out, "%c%c%c", r, g, b);
    }
  }
  //  fprintf(out, " ");
  fclose(out);
}

void skip_line(FILE *stream) {
  //printf("Eating line\n");
  int current = getc(stream);
  //printf("skip_line:current = %c(%d)\n",current,current);
  while (current != '\n') {
    current = getc(stream);
    //printf("skip_line:inside current = %c(%d)\n",current,current);
  }
  ungetc(current,stream);
}

void eat_whitespace(FILE *stream) {
  int current = getc(stream);
  //printf("current = %c(%d)\n",current,current);
  while (current == '\n' || current == ' ' ||
	 current == '\t' || current == '#') {
    //printf("current2 = %c(%d)\n",current,current);
    if (current == 35) {
      //printf("Inside if\n");
      skip_line(stream);
    }
    current = getc(stream);
    //printf("inside current = %c(%d)\n",current,current);
  }
  ungetc(current,stream);
}

void read_ppm(Color ** & buffer, int & width, int & height, char * filename) {
  int x, y;
  int depth;
  unsigned char r, g, b;
  FILE *in;

  
  in = fopen(filename, "r");
  if (!in) {
    fprintf(stderr, "Can't open input file %s\n", filename);
    exit(1);
  }
  else {
    char junk[10];
    // load in the file
    fscanf (in, "%s", junk);
    eat_whitespace(in);
    //printf("junk = %s\n",junk);
    fscanf (in, "%d", &width);
    eat_whitespace(in);
    //printf("width = %d\n",width);
    fscanf (in, "%d", &height);
    eat_whitespace(in);
    //printf("height = %d\n",height);
    fscanf (in, "%d", &depth);

    // using the width and height move the file pointer to the start of the
    // data rather than making sure that you get there by parsing stuff.
    fseek(in, -width*height*3, SEEK_END);
    
    // allocate memory
    buffer = (Color**) calloc(width, sizeof(Color*));
    Color* row = (Color*) calloc(width * height, sizeof(Color));
    for (int k = 0; k < width; k++) {
      buffer[k] = row;
      row += width;
    }

    printf("Reading image of (%d,%d)\n",width,height);
    
    // load in all the colors
    for (y = 0; y < height; y++) {
      for (x = 0; x < width; x++) {
	
	fscanf (in, "%c%c%c", &r, &g, &b);
	buffer[x][y] = Color((float)r / 255, (float)g / 255, (float)b / 255);
      }
    }
    fclose(in);
  }
}


///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
// main code

void matrix_test() {
  int n = 4;
  L_Matrix L(n,n);
  int count = 0;
#if 1
  for(L_Matrix::iterator i = L.begin(); i != L.end(); i++)
    for(L_Matrix::Row::iterator j = (*i).begin(); j != (*i).end(); j++)
      *j = count++;
#else
  for (int x = 0; x < L.nrows(); x++)
    for (int y = 0; y < L.ncols(); y++) {
      rows(L)[x][y] = count++;
    }
#endif
  print_all_matrix(L);

  ColorV F(n),P(n);
  for (ColorV::iterator i = F.begin(); i != F.end(); i++) {
    *i = Color(1,0.5,0.1);
    cout << *i << endl;
  }
  print_vector(F);
}

int main(int args, char** argv) {
  matrix_test();
  return 0;
  if (args < 2) {
    printf("ppmresize [ppm file] [optional output file]\n");
    exit(0);
  }
  Color ** buffer;
  int width, height;
  read_ppm(buffer,width,height,argv[1]);
  char *rtrtfile;
  if (args >= 3) {
    rtrtfile = argv[2];
  } else {
    rtrtfile = (char*)calloc(strlen(argv[1])+4,sizeof(char));
    strcpy(rtrtfile, argv[1]);
    rtrtfile[strlen(argv[1])-4] = '\0';
    strcat(rtrtfile,"_new.ppm");
  }
    
  printf("Reading from %s and writing to %s\n",argv[1],rtrtfile);
  write_ppm(buffer,width,height,rtrtfile);

  return 0;
}
