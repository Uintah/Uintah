//////////////////////////////////////////////////////////////////////
// Bmp.cpp - Read and Write Windows BMP images.
// Ripped off from XV v.3.10.
// Modified by Dave McAllister, 1999.

#include <Packages/Remote/Tools/Image/Quant.h>

#include <iostream>
#include <strings.h>
using namespace std;

#include <stdio.h>
#include <stdlib.h>

namespace Remote {
//----------------------------------------------------------------------
#if 0

typedef unsigned char byte;

#define BI_RGB 0
#define BI_RLE8 1
#define BI_RLE4 2

#define WIN_OS2_OLD 12
#define WIN_NEW 40
#define OS2_NEW 64

#define FERROR(fp) (ferror(fp) || feof(fp))

static long filesize;

/*******************************************/
static inline int bmpError(char *fname, char *st)
{
	cerr << fname << ": " << st << endl;
	return 0;
}

/*******************************************/
static inline unsigned int getshort(FILE *fp)
{
	int c, c1;
	c = getc(fp); c1 = getc(fp);
	return ((unsigned int) c) + (((unsigned int) c1) << 8);
}

/*******************************************/
static inline unsigned int getint(FILE *fp)
{
	int c, c1, c2, c3;
	c = getc(fp); c1 = getc(fp); c2 = getc(fp); c3 = getc(fp);
	return ((unsigned int) c) +
		(((unsigned int) c1) << 8) +
		(((unsigned int) c2) << 16) +
		(((unsigned int) c3) << 24);
}

/*******************************************/
static inline void putshort(FILE *fp, int i)
{
	int c, c1;
	
	c = ((unsigned int ) i) & 0xff; c1 = (((unsigned int) i)>>8) & 0xff;
	putc(c, fp); putc(c1,fp);
}

/*******************************************/
static inline void putint(FILE *fp, int i)
{
	int c, c1, c2, c3;
	c = ((unsigned int ) i) & 0xff;
	c1 = (((unsigned int) i)>>8) & 0xff;
	c2 = (((unsigned int) i)>>16) & 0xff;
	c3 = (((unsigned int) i)>>24) & 0xff;
	
	putc(c, fp); putc(c1,fp); putc(c2,fp); putc(c3,fp);
}

/*******************************************/
static int loadBMP1(FILE *fp, byte *pic8, unsigned int w, unsigned int h)
{
	int i,j,c,bitnum,padw;
	byte *pp;
	
	c = 0;
	padw = ((w + 31)/32) * 32; /* 'w', padded to be a multiple of 32 */
	
	for (i=h-1; i>=0; i--) {
		pp = pic8 + (i * w);

		for (j=bitnum=0; j<padw; j++,bitnum++) {
			if ((bitnum&7) == 0) { /* read the next byte */
				c = getc(fp);
				bitnum = 0;
			}
			
			if (j<w) {
				*pp++ = (c & 0x80) ? 1 : 0;
				c <<= 1;
			}
		}
		if (FERROR(fp)) break;
	}
	
	return (FERROR(fp));
}

/*******************************************/
static int loadBMP4(FILE *fp, byte *pic8, unsigned int w, unsigned int h, unsigned int comp)
{
	int i,j,c,c1,x,y,nybnum,padw,rv;
	byte *pp;
	
	rv = 0;
	c = c1 = 0;
	
	if (comp == BI_RGB) { /* read uncompressed data */
		padw = ((w + 7)/8) * 8; /* 'w' padded to a multiple of 8pix (32 bits) */
		
		for (i=h-1; i>=0; i--) {
			pp = pic8 + (i * w);
			if ((i&0x3f)==0) WaitCursor();
			
			for (j=nybnum=0; j<padw; j++,nybnum++) {
				if ((nybnum & 1) == 0) { /* read next byte */
					c = getc(fp);
					nybnum = 0;
				}
				
				if (j<w) {
					*pp++ = (c & 0xf0) >> 4;
					c <<= 4;
				}
			}
			if (FERROR(fp)) break;
		}
	}
	
	else if (comp == BI_RLE4) { /* read RLE4 compressed data */
		x = y = 0;
		pp = pic8 + x + (h-y-1)*w;
		
		while (y<h) {
			c = getc(fp); if (c == EOF) { rv = 1; break; }
			
			if (c) { /* encoded mode */
				c1 = getc(fp);
				for (i=0; i<c; i++,x++,pp++)
					*pp = (i&1) ? (c1 & 0x0f) : ((c1>>4)&0x0f);
			}
			
			else { /* c==0x00 : escape codes */
				c = getc(fp); if (c == EOF) { rv = 1; break; }
				
				if (c == 0x00) { /* end of line */
					x=0; y++; pp = pic8 + x + (h-y-1)*w;
				}
				
				else if (c == 0x01) break; /* end of pic8 */
				
				else if (c == 0x02) { /* delta */
					c = getc(fp); x += c;
					c = getc(fp); y += c;
					pp = pic8 + x + (h-y-1)*w;
				}
				
				else { /* absolute mode */
					for (i=0; i<c; i++, x++, pp++) {
						if ((i&1) == 0) c1 = getc(fp);
						*pp = (i&1) ? (c1 & 0x0f) : ((c1>>4)&0x0f);
					}
					
					if (((c&3)==1) || ((c&3)==2)) getc(fp); /* read pad byte */
				}
			} /* escape processing */
			if (FERROR(fp)) break;
		} /* while */
	}
	
	else {
		fprintf(stderr,"unknown BMP compression type 0x%0x\n", comp);
	}
	
	if (FERROR(fp)) rv = 1;
	return rv;
}

/*******************************************/
static int loadBMP8(FILE *fp, byte *pic8, unsigned int w, unsigned int h, unsigned int comp)
{
	int i,j,c,c1,padw,x,y,rv;
	byte *pp;
	
	rv = 0;
	
	if (comp == BI_RGB) { /* read uncompressed data */
		padw = ((w + 3)/4) * 4; /* 'w' padded to a multiple of 4pix (32 bits) */
		
		for (i=h-1; i>=0; i--) {
			pp = pic8 + (i * w);
			if ((i&0x3f)==0) WaitCursor();
			
			for (j=0; j<padw; j++) {
				c = getc(fp); if (c==EOF) rv = 1;
				if (j<w) *pp++ = c;
			}
			if (FERROR(fp)) break;
		}
	}
	
	else if (comp == BI_RLE8) { /* read RLE8 compressed data */
		x = y = 0;
		pp = pic8 + x + (h-y-1)*w;
		
		while (y<h) {
			c = getc(fp); if (c == EOF) { rv = 1; break; }
			
			if (c) { /* encoded mode */
				c1 = getc(fp);
				for (i=0; i<c; i++,x++,pp++) *pp = c1;
			}
			
			else { /* c==0x00 : escape codes */
				c = getc(fp); if (c == EOF) { rv = 1; break; }
				
				if (c == 0x00) { /* end of line */
					x=0; y++; pp = pic8 + x + (h-y-1)*w;
				}
				
				else if (c == 0x01) break; /* end of pic8 */
				
				else if (c == 0x02) { /* delta */
					c = getc(fp); x += c;
					c = getc(fp); y += c;
					pp = pic8 + x + (h-y-1)*w;
				}
				
				else { /* absolute mode */
					for (i=0; i<c; i++, x++, pp++) {
						c1 = getc(fp);
						*pp = c1;
					}
					
					if (c & 1) getc(fp); /* odd length run: read an extra pad byte */
				}
			} /* escape processing */
			if (FERROR(fp)) break;
		} /* while */
	}
	
	else {
		fprintf(stderr,"unknown BMP compression type 0x%0x\n", comp);
	}
	
	if (FERROR(fp)) rv = 1;
	return rv;
}

/*******************************************/
static int loadBMP24(FILE *fp, byte *pic24, unsigned int w, unsigned int h)
{
	int i,j,padb,rv;
	byte *pp;
	
	rv = 0;
	
	padb = (4 - ((w*3) % 4)) & 0x03; /* # of pad bytes to read at EOscanline */
	
	for (i=h-1; i>=0; i--) {
		pp = pic24 + (i * w * 3);
		if ((i&0x3f)==0) WaitCursor();
		
		for (j=0; j<w; j++) {
			pp[2] = getc(fp); /* blue */
			pp[1] = getc(fp); /* green */
			pp[0] = getc(fp); /* red */
			pp += 3;
		}
		
		for (j=0; j<padb; j++) getc(fp);
		
		rv = (FERROR(fp));
		if (rv) break;
	}
	
	return rv;
}

/*******************************************/
int LoadBMP(char *fname, PICINFO *pinfo)
{
	FILE *fp;
	int i, c, c1, rv;
	unsigned int bfSize, bfOffBits, biSize, biWidth, biHeight, biPlanes;
	unsigned int biBitCount, biCompression, biSizeImage, biXPelsPerMeter;
	unsigned int biYPelsPerMeter, biClrUsed, biClrImportant;
	int bPad;
	char *cmpstr;
	byte *pic24, *pic8;
	char buf[512], *bname;
	
	/* returns '1' on success */
	
	pic8 = pic24 = NULL;
	bname = BaseName(fname);
	
	fp = fopen(fname,"r");
	if (!fp) return (bmpError(bname, "couldn't open file"));
	
	fseek(fp, 0L, 2); /* figure out the file size */
	filesize = ftell(fp);
	fseek(fp, 0L, 0);
	
	/* read the file type (first two bytes) */
	c = getc(fp); c1 = getc(fp);
	if (c!='B' || c1!='M') { bmpError(bname,"file type != 'BM'"); goto ERROR; }
	
	bfSize = getint(fp);
	getshort(fp); /* reserved and ignored */
	getshort(fp);
	bfOffBits = getint(fp);
	
	biSize = getint(fp);
	
	if (biSize == WIN_NEW || biSize == OS2_NEW) {
		biWidth = getint(fp);
		biHeight = getint(fp);
		biPlanes = getshort(fp);
		biBitCount = getshort(fp);
		biCompression = getint(fp);
		biSizeImage = getint(fp);
		biXPelsPerMeter = getint(fp);
		biYPelsPerMeter = getint(fp);
		biClrUsed = getint(fp);
		biClrImportant = getint(fp);
	}
	
	else { /* old bitmap format */
		biWidth = getshort(fp); /* Types have changed ! */
		biHeight = getshort(fp);
		biPlanes = getshort(fp);
		biBitCount = getshort(fp);
		
		/* Not in old versions so have to compute them*/
		biSizeImage = (((biPlanes * biBitCount*biWidth)+31)/32)*4*biHeight;
		
		biCompression = BI_RGB;
		biXPelsPerMeter = biYPelsPerMeter = 0;
		biClrUsed = biClrImportant = 0;
	}
	
	if (DEBUG>1) {
		fprintf(stderr,"\nLoadBMP:\tbfSize=%d, bfOffBits=%d\n",bfSize,bfOffBits);
		fprintf(stderr,"\t\tbiSize=%d, biWidth=%d, biHeight=%d, biPlanes=%d\n",
			biSize, biWidth, biHeight, biPlanes);
		fprintf(stderr,"\t\tbiBitCount=%d, biCompression=%d, biSizeImage=%d\n",
			biBitCount, biCompression, biSizeImage);
		fprintf(stderr,"\t\tbiX,YPelsPerMeter=%d,%d biClrUsed=%d, biClrImp=%d\n",
			biXPelsPerMeter, biYPelsPerMeter, biClrUsed, biClrImportant);
	}
	
	if (FERROR(fp)) { bmpError(bname,"EOF reached in file header"); goto ERROR; }
	
	/* error checking */
	if ((biBitCount!=1 && biBitCount!=4 && biBitCount!=8 && biBitCount!=24) ||
		biPlanes!=1 || biCompression>BI_RLE4) {
		
		sprintf(buf,"Bogus BMP File! (bitCount=%d, Planes=%d, Compression=%d)",
			biBitCount, biPlanes, biCompression);
		
		bmpError(bname, buf);
		goto ERROR;
	}
	
	if (((biBitCount==1 || biBitCount==24) && biCompression != BI_RGB) ||
		(biBitCount==4 && biCompression==BI_RLE8) ||
		(biBitCount==8 && biCompression==BI_RLE4)) {
		
		sprintf(buf,"Bogus BMP File! (bitCount=%d, Compression=%d)",
			biBitCount, biCompression);
		
		bmpError(bname, buf);
		goto ERROR;
	}
	
	bPad = 0;
	if (biSize != WIN_OS2_OLD) {
		/* skip ahead to colormap, using biSize */
		c = biSize - 40; /* 40 bytes read from biSize to biClrImportant */
		for (i=0; i<c; i++) getc(fp);
		
		bPad = bfOffBits - (biSize + 14);
	}
	
	/* load up colormap, if any */
	if (biBitCount!=24) {
		int i, cmaplen;
		
		cmaplen = (biClrUsed) ? biClrUsed : 1 << biBitCount;
		for (i=0; i<cmaplen; i++) {
			pinfo->b[i] = getc(fp);
			pinfo->g[i] = getc(fp);
			pinfo->r[i] = getc(fp);
			if (biSize != WIN_OS2_OLD) {
				getc(fp);
				bPad -= 4;
			}
		}
		
		if (FERROR(fp))
		{ bmpError(bname,"EOF reached in BMP colormap"); goto ERROR; }
		
		if (DEBUG>1) {
			fprintf(stderr,"LoadBMP: BMP colormap: (RGB order)\n");
			for (i=0; i<cmaplen; i++) {
				fprintf(stderr,"%02x%02x%02x ", pinfo->r[i],pinfo->g[i],pinfo->b[i]);
			}
			fprintf(stderr,"\n\n");
		}
	}
	
	if (biSize != WIN_OS2_OLD) {
	/* Waste any unused bytes between the colour map (if present)
		and the start of the actual bitmap data. */
		
		while (bPad > 0) {
			(void) getc(fp);
			bPad--;
		}
	}
	
	/* create pic8 or pic24 */
	
	if (biBitCount==24) {
		pic24 = (byte *) calloc((size_t) biWidth * biHeight * 3, (size_t) 1);
		if (!pic24) return (bmpError(bname, "couldn't malloc 'pic24'"));
	}
	else {
		pic8 = (byte *) calloc((size_t) biWidth * biHeight, (size_t) 1);
		if (!pic8) return(bmpError(bname, "couldn't malloc 'pic8'"));
	}
	
	WaitCursor();
	
	/* load up the image */
	if (biBitCount == 1) rv = loadBMP1(fp,pic8,biWidth,biHeight);
	else if (biBitCount == 4) rv = loadBMP4(fp,pic8,biWidth,biHeight,
		biCompression);
	else if (biBitCount == 8) rv = loadBMP8(fp,pic8,biWidth,biHeight,
		biCompression);
	else rv = loadBMP24(fp,pic24,biWidth,biHeight);
	
	if (rv) bmpError(bname, "File appears truncated. Winging it.\n");
	
	fclose(fp);
	
	if (biBitCount == 24) {
		pinfo->pic = pic24;
		pinfo->type = PIC24;
	}
	else {
		pinfo->pic = pic8;
		pinfo->type = PIC8;
	}
	
	cmpstr = "";
	if (biCompression == BI_RLE4) cmpstr = ", RLE4 compressed";
	else if (biCompression == BI_RLE8) cmpstr = ", RLE8 compressed";
	
	pinfo->w = biWidth; pinfo->h = biHeight;
	pinfo->normw = pinfo->w; pinfo->normh = pinfo->h;
	pinfo->frmType = F_BMP;
	pinfo->colType = F_FULLCOLOR;
	
	sprintf(pinfo->fullInfo, "%sBMP, %d bit%s per pixel%s. (%ld bytes)",
		((biSize==WIN_OS2_OLD) ? "Old OS/2 " :
	(biSize==WIN_NEW) ? "Windows " : ""),
		biBitCount, (biBitCount == 1) ? "" : "s",
		cmpstr, filesize);
	
	sprintf(pinfo->shrtInfo, "%dx%d BMP.", biWidth, biHeight);
	pinfo->comment = (char *) NULL;
	
	return 1;
	
ERROR:
	fclose(fp);
	return 0;
}

/*******************************************/
static void writeBMP1(FILE *fp, byte *pic8, int w, int h)
{
	int i,j,c,bitnum,padw;
	byte *pp;
	
	padw = ((w + 31)/32) * 32; /* 'w', padded to be a multiple of 32 */
	
	for (i=h-1; i>=0; i--) {
		pp = pic8 + (i * w);
		if ((i&0x3f)==0) WaitCursor();
		
		for (j=bitnum=c=0; j<=padw; j++,bitnum++) {
			if (bitnum == 8) { /* write the next byte */
				putc(c,fp);
				bitnum = c = 0;
			}
			
			c <<= 1;
			
			if (j<w) {
				c |= (pc2nc[*pp++] & 0x01);
			}
		}
	}
}

/*******************************************/
static void writeBMP4(FILE *fp, byte *pic8, int w, int h)
{
	int i,j,c,nybnum,padw;
	byte *pp;
	
	padw = ((w + 7)/8) * 8; /* 'w' padded to a multiple of 8pix (32 bits) */
	
	for (i=h-1; i>=0; i--) {
		pp = pic8 + (i * w);
		if ((i&0x3f)==0) WaitCursor();
		
		for (j=nybnum=c=0; j<=padw; j++,nybnum++) {
			if (nybnum == 2) { /* write next byte */
				putc((c&0xff), fp);
				nybnum = c = 0;
			}
			
			c <<= 4;
			
			if (j<w) {
				c |= (pc2nc[*pp] & 0x0f);
				pp++;
			}
		}
	}
}

/*******************************************/
static void writeBMP8(FILE *fp, byte *pic8, int w, int h)
{
	int i,j,c,padw;
	byte *pp;
	
	padw = ((w + 3)/4) * 4; /* 'w' padded to a multiple of 4pix (32 bits) */
	
	for (i=h-1; i>=0; i--) {
		pp = pic8 + (i * w);
		if ((i&0x3f)==0) WaitCursor();
		
		for (j=0; j<w; j++) putc(pc2nc[*pp++], fp);
		for ( ; j<padw; j++) putc(0, fp);
	}
}

/*******************************************/
static void writeBMP24(FILE *fp, byte *pic24, int w, int h)
{
	int i,j,c,padb;
	byte *pp;
	
	padb = (4 - ((w*3) % 4)) & 0x03; /* # of pad bytes to write at EOscanline */
	
	for (i=h-1; i>=0; i--) {
		pp = pic24 + (i * w * 3);
		if ((i&0x3f)==0) WaitCursor();
		
		for (j=0; j<w; j++) {
			putc(pp[2], fp);
			putc(pp[1], fp);
			putc(pp[0], fp);
			pp += 3;
		}
		
		for (j=0; j<padb; j++) putc(0, fp);
	}
}

/*******************************************/
int WriteBMP(FILE *fp, byte *pic824, int ptype, int w, int h,
			 byte *rmap, byte *gmap, byte *bmap,
			 int numcols, int colorstyle)
{
/*
* if PIC8, and colorstyle == F_FULLCOLOR, F_GREYSCALE, or F_REDUCED,
* the program writes an uncompressed 4- or 8-bit image (depending on
* the value of numcols)
*
* if PIC24, and colorstyle == F_FULLCOLOR, program writes an uncompressed
* 24-bit image
* if PIC24 and colorstyle = F_GREYSCALE, program writes an uncompressed
* 8-bit image
* note that PIC24 and F_BWDITHER/F_REDUCED won't happen
*
* if colorstyle == F_BWDITHER, it writes a 1-bit image
*
	*/
	
	int i,j, nc, nbits, bperlin, cmaplen;
	byte *graypic, *sp, *dp, graymap[256];
	
	nc = nbits = cmaplen = 0;
	graypic = NULL;
	
	if (ptype == PIC24 && colorstyle == F_GREYSCALE) {
	/* generate a faked 8-bit per pixel image with a grayscale cmap,
		so that it can just fall through existing 8-bit code */
		
		graypic = (byte *) malloc((size_t) w*h);
		if (!graypic) FatalError("unable to malloc in WriteBMP()");
		
		for (i=0,sp=pic824,dp=graypic; i<w*h; i++,sp+=3, dp++) {
			*dp = MONO(sp[0],sp[1],sp[2]);
		}
		
		for (i=0; i<256; i++) graymap[i] = i;
		rmap = gmap = bmap = graymap;
		numcols = 256;
		ptype = PIC8;
		
		pic824 = graypic;
	}
	
	if (ptype == PIC24) { /* is F_FULLCOLOR */
		nbits = 24;
		cmaplen = 0;
		nc = 0;
	}
	
	else if (ptype == PIC8) {
	/* we may have duplicate colors in the colormap, and we'd prefer not to.
	* build r1,g1,b1 (a contiguous, minimum set colormap), and pc2nc[], a
	* array that maps 'pic8' values (0-numcols) into corresponding values
	* in the r1,g1,b1 colormaps (0-nc)
		*/
		
		for (i=0; i<256; i++) { pc2nc[i] = r1[i] = g1[i] = b1[i] = 0; }
		
		nc = 0;
		for (i=0; i<numcols; i++) {
			/* see if color #i is a duplicate */
			for (j=0; j<i; j++) {
				if (rmap[i] == rmap[j] && gmap[i] == gmap[j] &&
					bmap[i] == bmap[j]) break;
			}
			
			if (j==i) { /* wasn't found */
				pc2nc[i] = nc;
				r1[nc] = rmap[i];
				g1[nc] = gmap[i];
				b1[nc] = bmap[i];
				nc++;
			}
			else pc2nc[i] = pc2nc[j];
		}
		
		/* determine how many bits per pixel we'll be writing */
		if (colorstyle == F_BWDITHER || nc <= 2) nbits = 1;
		else if (nc<=16) nbits = 4;
		else nbits = 8;
		
		cmaplen = 1<<nbits; /* # of entries in cmap */
	}
	
	bperlin = ((w * nbits + 31) / 32) * 4; /* # bytes written per line */
	
	putc('B', fp); putc('M', fp); /* BMP file magic number */
	
	/* compute filesize and write it */
	i = 14 + /* size of bitmap file header */
		40 + /* size of bitmap info header */
		(nc * 4) + /* size of colormap */
		bperlin * h; /* size of image data */
	
	putint(fp, i);
	putshort(fp, 0); /* reserved1 */
	putshort(fp, 0); /* reserved2 */
	putint(fp, 14 + 40 + (nc * 4)); /* offset from BOfile to BObitmap */
	
	putint(fp, 40); /* biSize: size of bitmap info header */
	putint(fp, w); /* biWidth */
	putint(fp, h); /* biHeight */
	putshort(fp, 1); /* biPlanes: must be '1' */
	putshort(fp, nbits); /* biBitCount: 1,4,8, or 24 */
	putint(fp, BI_RGB); /* biCompression: BI_RGB, BI_RLE8 or BI_RLE4 */
	putint(fp, bperlin*h); /* biSizeImage: size of raw image data */
	putint(fp, 75 * 39); /* biXPelsPerMeter: (75dpi * 39" per meter) */
	putint(fp, 75 * 39); /* biYPelsPerMeter: (75dpi * 39" per meter) */
	putint(fp, nc); /* biClrUsed: # of colors used in cmap */
	putint(fp, nc); /* biClrImportant: same as above */
	
	/* write out the colormap */
	for (i=0; i<nc; i++) {
		if (colorstyle == F_GREYSCALE) {
			j = MONO(r1[i],g1[i],b1[i]);
			putc(j,fp); putc(j,fp); putc(j,fp); putc(0,fp);
		}
		else {
			putc(b1[i],fp);
			putc(g1[i],fp);
			putc(r1[i],fp);
			putc(0,fp);
		}
	}
	
	/* write out the image */
	if (nbits == 1) writeBMP1 (fp, pic824, w, h);
	else if (nbits == 4) writeBMP4 (fp, pic824, w, h);
	else if (nbits == 8) writeBMP8 (fp, pic824, w, h);
	else if (nbits == 24) writeBMP24(fp, pic824, w, h);
	
	if (graypic) free(graypic);
	
#ifndef VMS
	if (FERROR(fp)) return -1;
#else
	if (!FERROR(fp)) return -1;
#endif
	
	return 0;
}

#endif
} // End namespace Remote


