/*
 *	img_seek, img_write, img_read, img_optseek -
 *
 *				Paul Haeberli - 1984
 *
 */
#include	<stdio.h>
#include	"imagelib.h"

unsigned long img_optseek(IMAGE *image, unsigned long offset);

unsigned long img_seek(IMAGE *image, unsigned int y, unsigned int z)
{
    if(img_badrow(image,y,z)) {
	i_errhdlr("img_seek: row number out of range\n");
	return EOF;
    }
    image->x = 0;
    image->y = y;
    image->z = z;
    if(ISVERBATIM(image->type)) {
	switch(image->dim) {
	    case 1:
		return img_optseek(image, 512L);
	    case 2: 
		return img_optseek(image,512L+(y*image->xsize)*BPP(image->type));
	    case 3: 
		return img_optseek(image,
		    512L+(y*image->xsize+z*image->xsize*image->ysize)*
							BPP(image->type));
	    default:
		i_errhdlr("img_seek: weird dim\n");
		break;
	}
    } else if(ISRLE(image->type)) {
	switch(image->dim) {
	    case 1:
		return img_optseek(image, image->rowstart[0]);
	    case 2: 
		return img_optseek(image, image->rowstart[y]);
	    case 3: 
		return img_optseek(image, image->rowstart[y+z*image->ysize]);
	    default:
		i_errhdlr("img_seek: weird dim\n");
		break;
	}
    } else 
	i_errhdlr("img_seek: weird image type\n");
    return((unsigned long)-1);
}

int img_badrow(IMAGE *image, unsigned int y, unsigned int z)
{
    if(y>=image->ysize || z>=image->zsize)
	return 1;
    else
        return 0;
}

int img_write(IMAGE *image, char *buffer,int count)
{
    int retval;

    retval =  write(image->file,buffer,count);
    if(retval == count) 
	image->offset += count;
    else
	image->offset = -1;
    return retval;
}

int img_read(IMAGE *image, char *buffer, int count)
{
    int retval;

    retval =  read(image->file,buffer,count);
    if(retval == count) 
	image->offset += count;
    else
	image->offset = -1;
    return retval;
}

unsigned long img_optseek(IMAGE *image, unsigned long offset)
{
    if(image->offset != offset) {
       image->offset = offset;
       return ((unsigned long) lseek(image->file,offset,0));
   }
   return offset;
}

