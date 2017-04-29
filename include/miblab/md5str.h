#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* ==========================================================================
 * Function:	md5str ( instr )
 * Purpose:	returns null-terminated character string containing
 *		md5 hash of instr (input string)
 * --------------------------------------------------------------------------
 * Arguments:	instr (I)	pointer to null-terminated char string
 *				containing input string whose md5 hash
 *				is desired
 * --------------------------------------------------------------------------
 * Returns:	( char * )	ptr to null-terminated 32-character
 *				md5 hash of instr
 * --------------------------------------------------------------------------
 * Notes:     o	Other md5 library functions are included below.
 *		They're all taken from Christophe Devine's code,
 *		which (as of 04-Aug-2004) is available from
 *		     http://www.cr0.net:8040/code/crypto/md5/
 * ======================================================================= */
/* --- #include "md5.h" --- */
#ifndef uint8
  #define uint8  unsigned char
#endif
#ifndef uint32
  #define uint32 unsigned long int
#endif
typedef struct
  { uint32 total[2];
    uint32 state[4];
    uint8 buffer[64];
  } md5_context;
void md5_starts( md5_context *ctx );
void md5_update( md5_context *ctx, uint8 *input, uint32 length );
void md5_finish( md5_context *ctx, uint8 digest[16] );
/* --- md5.h --- */
#define GET_UINT32(n,b,i)                       \
  { (n) = ( (uint32) (b)[(i)    ]       )       \
        | ( (uint32) (b)[(i) + 1] <<  8 )       \
        | ( (uint32) (b)[(i) + 2] << 16 )       \
        | ( (uint32) (b)[(i) + 3] << 24 ); }
#define PUT_UINT32(n,b,i)                       \
  { (b)[(i)    ] = (uint8) ( (n)       );       \
    (b)[(i) + 1] = (uint8) ( (n) >>  8 );       \
    (b)[(i) + 2] = (uint8) ( (n) >> 16 );       \
    (b)[(i) + 3] = (uint8) ( (n) >> 24 ); }
/* --- p-defines as functions --- */
void P1(uint32 *X,uint32 *a,uint32 b,uint32 c,uint32 d,int k,int s,uint32 t);
void P2(uint32 *X,uint32 *a,uint32 b,uint32 c,uint32 d,int k,int s,uint32 t);
void P3(uint32 *X,uint32 *a,uint32 b,uint32 c,uint32 d,int k,int s,uint32 t);
void P4(uint32 *X,uint32 *a,uint32 b,uint32 c,uint32 d,int k,int s,uint32 t);

/* --- entry point (this one little stub written by me)--- */
void md5str( char* outstr, char *instr );
