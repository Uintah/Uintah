//////////////////////////////////////////////////////////////////////
// Quant.cpp - Load and save GIF images.
// Modified by David K. McAllister, 1997-1999.
// Taken from XV v. 3.10, which was taken from xgif and others 1987-1999.

#include <Packages/Remote/Tools/Image/Quant.h>
#include <Packages/Remote/Tools/Util/Assert.h>

#include <iostream>
#include <string.h>
#include <strings.h>
using namespace std;

#include <stdio.h>
#include <stdlib.h>

namespace Remote {
//////////////////////////////////////////////////////////////////////
// Color Quantization Class
//////////////////////////////////////////////////////////////////////

static inline int DiffP(const pixel &a, const pixel &b)
{
	return (int(a.r)-int(b.r))*(int(a.r)-int(b.r)) +
		((int(a.g)-int(b.g))*(int(a.g)-int(b.g))) +
		(int(a.b)-int(b.b))*(int(a.b)-int(b.b));
}

// See if there are <= MaxCols unique colors.
// If so, return pic8.
bool Quantizer::TrivialSolution(int MaxCols, byte *pic8)
{
  int y;
	NumColors = 0;
	
	for(y=0; y<size && NumColors <= MaxCols; y++)
	{
		pixel &p = Pix[y];
		bool FoundIt = false;
		for(int i=0; i<NumColors; i++)
		{
			if(p == cmap[i])
			{
				FoundIt = true;
				pic8[y] = i;
				break;
			}
		}
		if(!FoundIt)
		{
			if(NumColors >= MaxCols)
				break;

			cmap[NumColors] = p;
			pic8[y] = NumColors++;
		}
	}
	
	return (y >= size);
}

// Reduce the color map by computing the number of unique colors.
// Also overwrite pic8 with the new indices.
void Quantizer::ReduceColorMap(byte *pic8)
{
	byte cnt[256];
	pixel cmap2[256];
	byte pc2nc[256];
	int i;
	int j;
	
	memset(cnt, 0, 256);
	memset(pc2nc, 0, 256);
	memset(cmap2, 0, sizeof(pixel) * 256);
	
	// Flag which colors were actually used.
	for(i=0; i<size; i++)
	{
		cnt[pic8[i]] = 1;
	}
	
	// Reduce color map by removing unused and duplicate colors.
	int nc = 0;
	for(i=0; i<NumColors; i++)
	{
		// See if color #i is already used.
		for(j=0; j<i; j++)
		{
			if(cmap[i] == cmap[j])
				break;
		}
		
		// If the color is unique and used, add it.
		if(j==i && cnt[i])
		{
			// Wasn't found.
			pc2nc[i] = nc;
			cmap2[nc] = cmap[i];
			nc++;
		}
		else
			pc2nc[i] = pc2nc[j];
	}
	
	// Replace the image with a new one.
	for(i=0; i<size; i++)
	{
		pic8[i] = pc2nc[pic8[i]];
	}
	
	NumColors = nc;

	memcpy(cmap, cmap2, sizeof(pixel) * 256);
}

void Quantizer::CentroidsToColorMap(int *Counts, int *R, int *G, int *B)
{
	if(NumColors < 256)
		memset(&Counts[NumColors], 0, sizeof(int) * (256 - NumColors));
	
	NumColors = 256;
	for(int i=0; i<NumColors; i++)
	{
		if(Counts[i] > 0)
		{
			cmap[i].r = R[i] / Counts[i];
			cmap[i].g = G[i] / Counts[i];
			cmap[i].b = B[i] / Counts[i];
		}
		else
		{
#ifdef SCI_DEBUG
			cerr << "R";
#endif
			cmap[i].r = rand() & 0xff;
			cmap[i].g = rand() & 0xff;
			cmap[i].b = rand() & 0xff;
		}
	}
}

// Give a color map, convert the 24-bit image to 8 bits. No dither.
// Counts is how many of each pixel there are.
// Returns error.
INT64 Quantizer::Image24to8(byte *pic8, int *Counts, int *R, int *G, int *B)
{
	INT64 Error = 0;
	
	memset(Counts, 0, sizeof(int) * NumColors);
	memset(R, 0, sizeof(int) * NumColors);
	memset(G, 0, sizeof(int) * NumColors);
	memset(B, 0, sizeof(int) * NumColors);
	
	// Set each pixel to closest color.
	for(int y=0; y<size; y++)
	{
		int BestDist = 0xfffffff, BestC = 0;
		for(int i=0; i<NumColors; i++)
		{
			int Dist = DiffP(Pix[y], cmap[i]);
			if(Dist<BestDist)
			{
				BestDist = Dist;
				BestC = i;
			}
		}
		
		pic8[y] = (byte) BestC;
		Error += BestDist;
		
		Counts[BestC]++;
		
		R[BestC] += Pix[y].r;
		G[BestC] += Pix[y].g;
		B[BestC] += Pix[y].b;
	}
	
	return Error;
}

#define STOP_EARLY (size/2)

// Given an initial colormap, refine it to reduce error.
INT64 Quantizer::RefineColorMap(byte *pic8)
{
	int *redBig = new int[256];
	int *greenBig = new int[256];
	int *blueBig = new int[256];
	
	// The number of pixels in each cluster.
	int *countBig = new int[256];
	
	memset(redBig, 0, sizeof(int) * 256);
	memset(greenBig, 0, sizeof(int) * 256);
	memset(blueBig, 0, sizeof(int) * 256);
	memset(countBig, 0, sizeof(int) * 256);
	
	// Compute best fit to that map.
	INT64 OldError = 0x7fffffff;
	INT64 Error = OldError - STOP_EARLY - 1;
	
	// XXX Not looping here speeds things up.
	//while(Error + STOP_EARLY < OldError)
	for(int iter = 0; iter < 2; iter++)
	{
		OldError = Error;
		Error = Image24to8(pic8, countBig, redBig, greenBig, blueBig);
		
		CentroidsToColorMap(countBig, redBig, greenBig, blueBig);
		
#ifdef SCI_DEBUG
		fprintf(stderr, "%03d Mapping Error = %lld\n", iter, Error);
#endif
	}
	
	delete [] redBig;
	delete [] greenBig;
	delete [] blueBig;
	delete [] countBig;
	
	return Error;
}

// Give a color map, convert the 24-bit image to 8 bits. No dither.
// Counts is how many of each pixel there are.
// Returns error.
INT64 Quantizer::Image24to8Fast(color_count *chash, int hsize, int *Counts,
								int *R, int *G, int *B)
{
	INT64 Error = 0;
	
	memset(Counts, 0, sizeof(int) * NumColors);
	memset(R, 0, sizeof(int) * NumColors);
	memset(G, 0, sizeof(int) * NumColors);
	memset(B, 0, sizeof(int) * NumColors);
	
	// Set each pixel to closest color.
	for(int y=0; y<hsize; y++)
	{
		int BestDist = 0xfffffff, BestC = 0;
		for(int i=0; i<NumColors; i++)
		{
			int Dist = DiffP(chash[y].color, cmap[i]);
			if(Dist<BestDist)
			{
				BestDist = Dist;
				BestC = i;
			}
		}
		
		Error += BestDist * chash[y].value;
		
		Counts[BestC] += chash[y].value;
		
		R[BestC] += chash[y].color.r * chash[y].value;
		G[BestC] += chash[y].color.g * chash[y].value;
		B[BestC] += chash[y].color.b * chash[y].value;
	}
	
	return Error;
}

// Given an initial colormap, refine it to reduce error.
INT64 Quantizer::RefineColorMapFast(color_count *chash, int hsize)
{
	int *redBig = new int[256];
	int *greenBig = new int[256];
	int *blueBig = new int[256];
	
	// The number of pixels in each cluster.
	int *countBig = new int[256];
	
	// Compute best fit to that map.
	INT64 OldError = 0x7fffffff;
	INT64 Error = OldError - STOP_EARLY - 1;
	
	int iter = 0;
	while(Error + STOP_EARLY < OldError)
	{
		OldError = Error;
		Error = Image24to8Fast(chash, hsize, countBig, redBig, greenBig, blueBig);
		
		CentroidsToColorMap(countBig, redBig, greenBig, blueBig);
		
		iter++;
#ifdef SCI_DEBUG
		fprintf(stderr, "%03d Fast Mapping Error = %lld\n", iter, Error);
#endif
	}
	
	delete [] redBig;
	delete [] greenBig;
	delete [] blueBig;
	delete [] countBig;
	
	return Error;
}

//////////////////////////////////////////////////////////////////////
// Median Cut Color Quantization Algorithm
//////////////////////////////////////////////////////////////////////

struct box
{
	int index;
	int colors;
	int sum;
};

/**********************************/
static int redcompare(const void *p1, const void *p2)
{
	return (int) ((color_count*)p1)->color.r -
		(int) ((color_count*)p2)->color.r;
}

/**********************************/
static int greencompare(const void *p1, const void *p2)
{
	return (int) ((color_count*)p1)->color.g -
		(int) ((color_count*)p2)->color.g;
}

/**********************************/
static int bluecompare(const void *p1, const void *p2)
{
	return (int) ((color_count*)p1)->color.b -
		(int) ((color_count*)p2)->color.b;
}

/**********************************/
static int sumcompare(const void *p1, const void *p2)
{
	return ((box *) p2)->sum - ((box *) p1)->sum;
}

// This keeps 6 bits.
//#define PMASK 0xfc
#define PMASK 0xf8

///////////////////////////////////////////////////////////////
#define PPM_ASSIGN(p,red,grn,blu) { p.r = red; p.g = grn; p.b = blu; }

/* Luminance macro, using only integer ops. Returns an int (*256) JHB */
#define PPM_LUMIN(p) (77 * p.r + 150 * p.g + 29 * p.b)

// This is 2^(6*3) plus a few.
//#define HASH_SIZE 262147
#define HASH_SIZE 32769

#define ppm_hashpixel(p) ((((int) p.r * 33023 + (int) p.g * 30013 + \
(int) p.b * 27011) & 0x7fffffff) % HASH_SIZE)

///////////////////////////////////////////////////////////////
color_count *Quantizer::ppm_computechash(int &num_colors)
{
	num_colors = 0;
	color_count *chash = new color_count[HASH_SIZE];
	memset(chash, 0, sizeof(color_count) * HASH_SIZE);
	
	// Go through the entire image, building a hash table of colors.
	for(int i=0; i<size; i++)
	{
		pixel pp = Pix[i];
		pp.r &= PMASK;
		pp.g &= PMASK;
		pp.b &= PMASK;
		
		int hash = ppm_hashpixel(pp);
		
		for(int j = hash; j != hash-1; j = (j+1) % HASH_SIZE)
		{
			if(chash[j].color == pp)
			{
				chash[j].value++;
				break;
			}
			else if(chash[j].value == 0)
			{
				chash[j].color.r = pp.r;
				chash[j].color.g = pp.g;
				chash[j].color.b = pp.b;
				chash[j].value = 1;
				num_colors++;
				break;
			}
		}
	}
	
	return chash;
}

/****************************************************************************
** Here is the fun part, the median-cut colormap generator. This is based
** on Paul Heckbert's paper "Color Image Quantization for Frame Buffer
** Display", SIGGRAPH '82 Proceedings, page 297.
*/
void Quantizer::mediancut(color_count* chv, int colors,
						  int sum, int maxval, int MaxColors)
{
	int bi, i;
	
	box *bv = new box[MaxColors];
	
	if(!bv) FatalError("unable to allocate in mediancut()");
	
	memset(cmap, 0, MaxColors * sizeof(pixel));
	
	// Set up the initial box.
	bv[0].index = 0;
	bv[0].colors = colors;
	bv[0].sum = sum;
	int boxes = 1;
	
	// Main loop: split boxes until we have enough.
	while (boxes < MaxColors)
	{
		int indx, clrs;
		int sm;
		int minr, maxr, ming, maxg, minb, maxb, v;
		int halfsum, lowersum;
		
		// Find the first splittable box.
		for(bi=0; bv[bi].colors<2 && bi<boxes; bi++)
			;
		if(bi == boxes)
			break; // No splittable boxes.
		
		indx = bv[bi].index;
		clrs = bv[bi].colors;
		sm = bv[bi].sum;
		
		// Go through the box finding the minimum and maximum of each
		// component - the boundaries of the box.
		minr = maxr = chv[indx].color.r;
		ming = maxg = chv[indx].color.g;
		minb = maxb = chv[indx].color.b;
		
		for(i=1; i<clrs; i++) {
			v = chv[indx + i].color.r;
			if(v < minr) minr = v;
			if(v > maxr) maxr = v;
			
			v = chv[indx + i].color.g;
			if(v < ming) ming = v;
			if(v > maxg) maxg = v;
			
			v = chv[indx + i].color.b;
			if(v < minb) minb = v;
			if(v > maxb) maxb = v;
		}
		
		/*
		** Find the largest dimension, and sort by that component. I have
		** included two methods for determining the "largest" dimension;
		** first by simply comparing the range in RGB space, and second
		** by transforming into luminosities before the comparison. We use
		** the luminosity version.
		*/
		{
			pixel p;
			int rl, gl, bl;
			
			PPM_ASSIGN(p, maxr - minr, 0, 0);
			rl = PPM_LUMIN(p);
			
			PPM_ASSIGN(p, 0, maxg - ming, 0);
			gl = PPM_LUMIN(p);
			
			PPM_ASSIGN(p, 0, 0, maxb - minb);
			bl = PPM_LUMIN(p);
			
			if(rl >= gl && rl >= bl)
				qsort((char*) &(chv[indx]), (size_t) clrs, sizeof(color_count),
				redcompare);
			else if(gl >= bl)
				qsort((char*) &(chv[indx]), (size_t) clrs, sizeof(color_count),
				greencompare);
			else
				qsort((char*) &(chv[indx]), (size_t) clrs, sizeof(color_count),
				bluecompare);
		}
		
		/*
		** Now find the median based on the counts, so that about half the
		** pixels (not colors, pixels) are in each subdivision.
		*/
		lowersum = chv[indx].value;
		halfsum = sm / 2;
		for(i=1; i<clrs-1; i++) {
			if(lowersum >= halfsum) break;
			lowersum += chv[indx + i].value;
		}
		
		/*
		** Split the box, and sort to bring the biggest boxes to the top.
		*/
		bv[bi].colors = i;
		bv[bi].sum = lowersum;
		bv[boxes].index = indx + i;
		bv[boxes].colors = clrs - i;
		bv[boxes].sum = sm - lowersum;
		++boxes;
		qsort((char*) bv, (size_t) boxes, sizeof(struct box), sumcompare);
	}
	
	/*
	** Ok, we've got enough boxes. Now choose a representative color for
	** each box. There are a number of possible ways to make this choice.
	** One would be to choose the center of the box; this ignores any structure
	** within the boxes. Another method would be to average all the colors in
	** the box - this is the method specified in Heckbert's paper. A third
	** method is to average all the pixels in the box. In other words, take a
	** weighted average of the colors. This is what we do.
	*/
	
	for(bi=0; bi<boxes; bi++) {
		int indx = bv[bi].index;
		int clrs = bv[bi].colors;
		INT64 r = 0, g = 0, b = 0, sum = 0;
		
		for(i=0; i<clrs; i++) {
			r += chv[indx + i].color.r * chv[indx + i].value;
			g += chv[indx + i].color.g * chv[indx + i].value;
			b += chv[indx + i].color.b * chv[indx + i].value;
			sum += chv[indx + i].value;
		}
		
		r = r / sum; if(r>maxval) r = maxval; /* avoid math errors */
		g = g / sum; if(g>maxval) g = maxval;
		b = b / sum; if(b>maxval) b = maxval;
		
		PPM_ASSIGN(cmap[bi], byte(r), byte(g), byte(b));
	}
	
	delete [] bv;
}

// Sort the table to put empty elements at the end.
void HashToHist(color_count *chash, int colors)
{
	int top = HASH_SIZE - 1;
	for(int i=0; i<top; i++)
	{
		if(chash[i].value == 0)
		{
			for( ; top > i; top--)
			{
				if(chash[top].value)
				{
					chash[i] = chash[top];
					chash[top].value = 0;
					break;
				}
			}
		}
	}
}

/****************************************************************************/
byte *Quantizer::Quant(byte *pic24, int sz, int MaxColors, bool IsGray)
{
	size = sz;
	
	byte *pic8 = new byte[size];

	if(IsGray)
	{
#ifdef SCI_DEBUG
		cerr << "Gray.\n";
#endif
		// For gray scale, just do it.
		memcpy(pic8, pic24, size);
		NumColors = MaxColors;

		for(int i=0; i<NumColors; i++)
		{
			cmap[i].r = cmap[i].g = cmap[i].b = i;
		}

		ReduceColorMap(pic8);
		
		return pic8;
	}
	
	Pix = (pixel *) pic24;
	byte maxval = 255;
	int colors;
	
	if(TrivialSolution(MaxColors, pic8))
		return pic8;
	
#ifdef SCI_DEBUG
	cerr << "Making histogram.\n";
#endif

	color_count *chash = ppm_computechash(colors);
	
#ifdef SCI_DEBUG
	cerr << colors << " unique colors found\n";
#endif
	
	// Convert the hash table to a histogram by moving used values to the top.
	HashToHist(chash, colors);
	
#ifdef SCI_DEBUG
	cerr << "Choosing " << MaxColors << " colors\n";
#endif
	
	// Apply median-cut to histogram, making the new colormap.
	mediancut(chash, colors, size, maxval, MaxColors);
	
#ifdef SCI_DEBUG
	cerr << "Finished median cut.\n";
#endif
	
	NumColors = MaxColors;
	// ReduceColorMap(pic8);
	
	RefineColorMapFast(chash, colors);
	RefineColorMap(pic8);

	delete [] chash;
	
	return pic8;
}
} // End namespace Remote


