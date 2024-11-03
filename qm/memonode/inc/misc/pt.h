#ifndef _H_PT
#define _H_PT

#ifdef DEF_INVERSE_BO
#define BO_BIG_ENDIAN	0	//컴파일 cpu가 big-endian이면 1, little-endian이면 0
#else
#define BO_BIG_ENDIAN	1
#endif
typedef signed char sytex;//k227so.arm machine에서 char가 default로 unsigned이므로 단일 문자만 부호문자를 따로 정의
typedef char bytex; // 1 byte
typedef short shortx; // 2 byte
typedef int intx; // 4 byte
typedef unsigned char ubytex;
typedef unsigned short ushortx;
typedef unsigned int uintx;
typedef float floatx;
typedef double doublex;

#ifdef OPT_WIN
typedef __int64 longx;
typedef unsigned __int64 unit;//8바이트 address용
#else
typedef long long longx;
typedef unsigned long long unit;//8바이트 address용
#endif

#define POSITIVE_INT_MAX	0x7fffffff
#define UNSIGNED_INT_MAX	0xffffffff
#define POSITIVE_LONG_MAX	0x7fffffffffffffff
#define UNSIGNED_LONG_MAX	0xffffffffffffffff

#ifdef OPT_ADDR64
#define MAX_SIZEX UNSIGNED_LONG_MAX
#define SMAX_SIZEX POSITIVE_LONG_MAX
#ifdef OPT_WIN
typedef __int64 decx;
typedef unsigned __int64 sizex;
typedef __int64 sockx;
#else
typedef long long decx;
typedef unsigned long long sizex;
typedef long long sockx;
#endif
#define CASTDEC() (intx)
#define CASTSIZE() (uintx)
#define CASTLONG2INT()			//k97so.
#define CASTINT2LONG()	(longx) //k97so.
typedef unit	sens_t_adr;
#else
#define MAX_SIZEX UNSIGNED_INT_MAX
#define SMAX_SIZEX POSITIVE_INT_MAX
#define CASTDEC()
#define CASTSIZE()
#define CASTLONG2INT()	(intx)
#define CASTINT2LONG()
typedef int decx;
typedef unsigned int sizex;
typedef int sockx;
#ifdef CPLUS_NEW_VER
#include <stddef.h>
#endif
typedef uintx	sens_t_adr;
#endif

#ifdef INTX64
#define CASTNUM() (intx)
typedef longx numx;
#define NUMX_MAX POSITIVE_LONG_MAX
#else 
#define CASTNUM()
typedef intx numx;
#define NUMX_MAX POSITIVE_INT_MAX
#endif

//netaddrx타입은 나중에 아피어드레스가 6바이트로 확장되면 6바이트 혹은 8바이트 사이즈로 한다.
typedef unsigned int netaddrx;

//#ifdef OPT_ADDR64
typedef unit divadx;	//포인터 어드레스 전용(diversion) 타입
typedef doublex unifltx;	//floating point 대표타입
//#else
//typedef unsigned int divadx;
//typedef floatx unifltx;
//#endif

#define UNITZ	(unit)0
#define PUZ	(unit *)0
#define nullc	'\0'
#define nullx NULL

#endif
