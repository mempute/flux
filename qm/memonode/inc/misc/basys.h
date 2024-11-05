#ifndef _H_BASYS
#define _H_BASYS

#include <stdio.h>
#include <stdlib.h>
#ifdef CPLUS_NEW_VER
#ifdef CPLUS_NEW_VER2
#include <string.h>
#else
#include <string>
#endif
#else
#include <string.h>
#endif

#ifdef OPT_WIN
#include <windows.h>
#else
#include <unistd.h>
#include <stdarg.h>
#endif


#endif

