//////////////////////////////////////////////////////////////////////
// Quant.h - Quantize color images to 256 colors for saving.
// Modified by David K. McAllister, 1997-1999.
// Taken from XV v. 3.10, which was taken from xgif and others 1987-1999.

#include <iostream>
#include <strings.h>

#include <stdio.h>
#include <stdlib.h>

namespace Remote {
using namespace std;
typedef unsigned char byte;
typedef long long INT64;

//////////////////////////////////////////////////////////////////////
// Color Quantization Class
//////////////////////////////////////////////////////////////////////

struct pixel
{
	byte r, g, b;

	inline bool operator==(const pixel &a) const
	{
		return a.r == r && a.g == g && a.b == b;
	}
};

// Color histogram stuff.
struct color_count
{
	pixel color;
	int value;
};

//////////////////////////////////////////////////////////////////////
// Median Cut Color Quantization Algorithm
// With Color Refinement by David K. McAllister
//////////////////////////////////////////////////////////////////////

struct Quantizer
{
	int size;
	int NumColors;
	pixel cmap[256];
	pixel *Pix;

	/**************************/
	byte *Quant(byte *pic24, int sz, int newcolors, bool IsGray = false);

private:
	//////////////////////////////
	inline void FatalError(const char *st)
	{
		cerr << st << endl;
		exit(1);
	}

	// See if there are fewer than MaxCols unique colors.
	// If so, return pic8.
	bool TrivialSolution(int MaxCols, byte *);

	// Reduce the color map by computing the number of unique colors.
	// Also overwrite pic8 with the new indices.
	void ReduceColorMap(byte *pic8);

	// Give a color map, convert the 24-bit image to 8 bits. No dither.
	// Counts is how many of each pixel there are.
	// Returns error.
	INT64 Image24to8(byte *pic8, int *Counts, int *R=NULL, int *G=NULL, int *B=NULL);

	INT64 RefineColorMap(byte *pic8);

	INT64 Image24to8Fast(color_count *chash, int hsize, int *Counts, int *R=NULL, int *G=NULL, int *B=NULL);
	INT64 RefineColorMapFast(color_count *chash, int hsize);

	// Set the color map to these centroids.
	void CentroidsToColorMap(int *Count, int *R, int *G, int *B);

	void mediancut(color_count* chv, int colors,
		int sum, int maxval, int newcolors);

	color_count* ppm_computechash(int &colors);
};

} // End namespace Remote


