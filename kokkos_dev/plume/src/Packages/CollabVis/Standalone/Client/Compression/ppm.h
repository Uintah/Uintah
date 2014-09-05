#ifndef _PPM_H_
#define _PPM_H_

/**
 *  Writes a ppm file to disk
 *
 * @param filename       Name of the file
 * @param buffer         Data to write
 * @param width          Width of buffer
 * @param height         Height of buffer
 */
extern void write_PPM( const char *filename, unsigned char * buffer,
		       int width, int height);
#endif
