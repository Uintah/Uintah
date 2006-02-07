#include <Compression/RLE.h>


namespace SemotusVisum {

const char * const
RLECompress::name = "RLE";

/* ------------------ RLE COMPRESSION ------------------------ */

/* Constructor. No operations necessary */
RLECompress::RLECompress()
{
}

/* Deconstructor. No operations necessary */
RLECompress::~RLECompress()
{
}

/* Compression routine. */
int
RLECompress::compress(DATA * input,
		      int width,
		      int height,
		      DATA ** output,
		      int bps,
		      int delta)
{
  int size = 0;
    
  /* Do some sanity checking */
  if (!input || !output)
    return -1;

  /* See if we need to allocate memory + safety factor for incompressible
     data*/
  if ( *output == NULL )
    *output = scinew DATA[ (int)(width * height * bps * 1.1) ];
  
  /* Do the compression! */
  size = encode( input, width*height*bps, output[0] );
  
  return size;
}

/* Decompression routine. */
int
RLECompress::decompress(DATA * input,
			int buffer_length,
			DATA ** output,
			int delta)
{ 
  int size = 0;
    
  /* Do some sanity checking */
  if (!input || !output)
    return -1;

  /* See if we need to allocate memory. */
  if (*output == NULL) {
    *output = allocateMemory(buffer_length * 256); //keep it safe.
    if (*output == NULL)
      return -1;
  }

  /* Do the decompression */
  size = decode(input,
		buffer_length,
		*output);
  
  return size;
}

/* Based on 8-bit RLE code by Shaun Case */
int
RLECompress::encode(DATA * input, int input_len, DATA * output)
{
  register int cur_char;
  unsigned short seq_len=0;    /* length of non-run sequence     */ 
  register unsigned short run_len = 0;  /* length of character run so far */
  int run_char=0;                         /* which char run is of           */
  int outputIndex = 0;
  int j;
  char seq[MAX_LEN];
  
  for ( int i = 0; i < input_len; i++ ) {
    cur_char = input[i];
    //printf("Processing character %c, index %d\n", cur_char,i );

    if (seq_len ==0)                /* haven't got a sequence yet   */
    {
      //printf("\tNo Sequence Yet\n");
      if (run_len == 0)           /* start a new run              */
      {
	//printf("\t\tStarting new run\n");
	run_char = cur_char;
	++run_len;
	continue;
      }

      if (run_char == cur_char) {   /* got another char in the run  */
	//printf("\t\tGot another char in run\n");
	if (++run_len == MAX_LEN)
	{
	  output[ outputIndex++ ] = MAX_RUN_HEADER;
	  output[ outputIndex++ ] = run_char;
	  
	  run_len = 0;
	  continue;
	}
      }
      
      /* got a different character than the run we were building,
	 so write out the run and start a new one of the new character. */
      if (run_len > 2){
	//printf("\t\tGot different char. Expecting %c\n", run_char );
	output[ outputIndex++ ] = RUN | run_len;
	output[ outputIndex++ ] = run_char;

	run_len  = 1;
	run_char = cur_char;
	continue;
      }

      /* run was only one or two chars, make a seq out of it instead       */
      //printf("\t\tMaking sequence of run\n");
      for (j = 0; j < run_len; j++);    /* copy 1 or 2 char run to seq[]   */
      {
	seq[seq_len] = run_char;
	++seq_len;
	if (seq_len == MAX_LEN)       /* if seq[] is full, write to disk */
	{
	  //printf("\t\t\tWriting full sequence.\n");
	  output[ outputIndex++ ] = MAX_SEQ_HEADER;
	    
	    for (int k = 0; k < seq_len; k++)
	      output[ outputIndex++ ] = seq[k];
	  
	  seq_len = 0;
	}
      }
      
      run_len = 0;

      seq[seq_len++] = cur_char;
      if (seq_len == MAX_LEN)        /* if seq[] is full, write to output */
      {
	//printf("\t\t\tWriting output\n");
	output[ outputIndex++ ] = MAX_SEQ_HEADER;

	for (int k = 0; k < seq_len; k++)
	  output[ outputIndex++ ] = seq[k];
	
	
	seq_len = 0;
      }
    }
    else    /* a sequence exists */
    {
      //printf("\tGot a sequence\n");
      if (run_len != 0)           /* if a run exists */
      {
	//printf("\t\tRun exists\n");
	if (cur_char == run_char )  /* add to run!  Yay.  */
	{
	  //printf("\t\t\tAdding to run\n");
	  ++run_len;
	  if (run_len == MAX_LEN)  /* if run is full */
	  {
	    /* write sequence that precedes run */
	    //printf("\t\t\t\tRun is full.\n");
	    output[ outputIndex++ ] = (SEQ | seq_len);

	    for (int k = 0; k < seq_len; k++)
	      output[ outputIndex++ ] = seq[k];

	    
	    /* write run                        */

	    output[ outputIndex++ ] = RUN | run_len;
	    output[ outputIndex++ ] = run_char;

	    /* and start out fresh              */
	    seq_len = run_len = 0;
	    
	  }  /* end write full run with existing sequence */

	  continue;

	}  /* end add to run for sequence exists */
	
	/* we couldn't add to the run, and a preceding sequence */
	/* exists, so write the sequence and the run, and       */
	/* try starting a new run with the current character.   */
	
	/* write sequence that precedes run */
	//printf("\t\t\tCouldn't add to run. Output index = %d, len=%d\n",outputIndex, seq_len );
	output[ outputIndex++ ] = SEQ | seq_len;

	for (int k = 0; k < seq_len; k++)
	  output[ outputIndex++ ] = seq[k];
	  

	
	/* write run                        */
	output[ outputIndex++ ] = RUN | run_len;
	output[ outputIndex++ ] = run_char;
	
	/* and start a new run w/ cur_char  */
	seq_len = 0;
	run_len = 1;
	run_char = cur_char;
	
	continue;
	
      }    /* end can't add to existing run, and preceding seq exists */

      /* no run exists, but a sequences does.  Try to create a run    */
      /* by looking at cur_char and the last char of the sequence.    */
      /* if that fails, add the char to the sequence.                 */
      /* if the sequence is full, write it to disk.  (Slightly non    */
      /* optimal; we could wait one more char.  A small thing to fix  */
      /* if someone gets the urge...                                  */
      
      if (seq[seq_len - 1] == cur_char)       /* if we can make a run */
      {
	//printf("\t\t\t\tTrying to make a run\n");
	run_char = cur_char;
	run_len = 2;
	--seq_len;
	continue;
      }
      
      /* couldn't make a run, add char to seq.  Maybe next time       */
      /* around...                                                    */
      //printf("\t\t\tAdding char to sequence\n");
      seq[seq_len++] = cur_char;
      
      if (seq_len == MAX_LEN) /* if the sequence is full, write out   */
      {
	output[ outputIndex++ ] = MAX_SEQ_HEADER;

	for (int k = 0; k < MAX_LEN; k++)
	  output[ outputIndex++ ] = seq[k];
	
	seq_len = 0;
      }
      
    }  /* end branch on sequence exists */

  } /* done with whole file */

  /* there may be stuff left that hasn't been written yet; if so, write it */
  
  if (seq_len != 0)  /* write sequence that precedes run */
  {
    output[ outputIndex++ ] = SEQ | seq_len;

    for (int k = 0; k < seq_len; k++)
      output[ outputIndex++ ] = seq[k];
  }
  
  if (run_len != 0)  /* write run */
  {
    output[ outputIndex++ ] = RUN | run_len;
    output[ outputIndex++ ] = run_char;
  }

  return outputIndex;
}

int
RLECompress::decode(DATA * input, int input_len, DATA * output)
{
  int packet_hdr;
  register int byte;
  register unsigned short length;
  int i = 0;
  int j;
  int outputIndex = 0;
  
  while ( i < input_len ) {
    packet_hdr = input[i++];
    length = MAX_LEN & packet_hdr;
    //printf("HDR = %d. Length = %d\n", packet_hdr, length );
    if (packet_hdr & RUN)  /* if it's a run... */
    {
      byte = input[i++];

      for ( j = 0; j < length; j++) {
	output[ outputIndex ] = byte;
	outputIndex ++;
      }
    }
    else /* it's a sequence */
      for (j = 0; j < length; j++) {
	output[ outputIndex ] = input[ i++ ];
	outputIndex++;
      }
  }

  return outputIndex;
}

}
