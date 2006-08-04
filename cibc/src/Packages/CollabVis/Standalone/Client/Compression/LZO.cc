#include <Compression/LZO.h>


namespace SemotusVisum {

const char * const
LZOCompress::name = "LZO";

/* ------------------ LZO COMPRESSION ------------------------ */

/* Constructor. Initialize the LZO library. */
LZOCompress::LZOCompress()
{
  if (lzo_init() != LZO_E_OK)
    printf("lzo_init() failed !!!\n");
}

/* Deconstructor. No operations necessary */
LZOCompress::~LZOCompress()
{
}

/* Compression routine. */
int
LZOCompress::compress(DATA * input,
		      int width,
		      int height,
		      DATA ** output,
		      int bps,
		      int delta)
{
  lzo_byte *wrkmem;
  lzo_byte *out;
  lzo_uint in_len;
  lzo_uint out_len;
  int ret;
  
  /* Do some sanity checking */
  if (!input || !output)
    return -1;

  /* See if we need to allocate memory */
  if (*output == NULL)
    {
      *output = allocateMemory(width * height);
      if (*output == NULL)
	return -1;
    }
  out = *output;
  wrkmem = /*lzo_malloc*/ scinew lzo_byte[LZO1X_1_MEM_COMPRESS];
  if (!wrkmem)
    return -1;

  //printf("Allocated %d bytes for output\n", width*height );fflush(stdout);
  
  /* Do the compression */
  in_len = width*height*sizeof(DATA)*bps;
  //printf("Input length: %d bytes\n", in_len );
  ret = lzo1x_1_compress(input,in_len,out,&out_len,wrkmem);
  if (ret != LZO_E_OK)
  {
    printf("Merde!\n\n\n\n");
    delete wrkmem;
    return -ret;
  }
  
  /* Free memory */
  delete wrkmem;
  return out_len;
}

/* Decompression routine. */
int
LZOCompress::decompress(DATA * input,
			int buffer_length,
			DATA ** output,
			int delta)
{
  lzo_byte *wrkmem;
  lzo_byte *out;
  lzo_uint out_len;
  int ret;
  
  /* Do some sanity checking */
  if (!input || !output)
    return -1;

  /* See if we need to allocate memory */
  if (*output == NULL)
    {
      *output = allocateMemory(buffer_length); // ARG - we need more!
      if (*output == NULL)
	return -1;
    }
  out = *output;
  wrkmem = /*lzo_malloc*/ scinew lzo_byte[LZO1X_1_MEM_COMPRESS];
  if (!wrkmem)
    return -1;

  /* Do the decompression */
  ret = lzo1x_decompress(input,buffer_length,out,&out_len,wrkmem);
  if (ret != LZO_E_OK)
    {
      delete wrkmem;
      return -ret;
    }

  /* Free memory */
  delete wrkmem;
  return out_len;
}

}
