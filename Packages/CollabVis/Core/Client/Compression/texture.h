/**
 * Read in an image file in SGI 'libimage' format
 * 	currently it's very simple minded and converts all images
 *      to RGBA8 regardless of the input format and returns the
 *	original number of components in the appropriate parameter.
 *    
 *     
 *	the components are converted as follows
 *		L    -> LLL 1.0
 *		LA   -> LLL A
 *		RGB  -> RGB 1.0
 *		RGBA -> RGB A
 *
 * @param name        Name of the file to read
 * @param width       Width of the file in pixels
 * @param height      Height of the file in pixels
 * @param components  Number of components (RGBA, etc).
 */
unsigned *
read_texture(const char *name, int *width, int *height, int *components);
