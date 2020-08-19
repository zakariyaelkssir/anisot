#ifndef _XMALLOC_C
#define _XMALLOC_C

#include <stdio.h>
#include <stdlib.h>


static void *xmalloc(size_t size)
{
	void *p = malloc(size);
	if (!p)
		exit(fprintf(stderr, "out of memory\n"));
	return p;
}
#endif
