#ifndef _H_XT
#define _H_XT

#include "misc/basys.h"
#include "misc/dbg.h"

#include <time.h>

#ifdef OPT_WIN

typedef HANDLE thrid;
typedef LPVOID thrarg;
typedef HANDLE	hmutex;
typedef HANDLE	hsignal;
typedef char FAR *pvirmemx;
typedef __time64_t timex64;

#define thrret DWORD

typedef intx socklenx;

struct timezonex {
	intx tz_minuteswest; // minutes W of Greenwich
	intx tz_dsttime;     // type of dst correction
};

#else

#include <pthread.h>
#if defined(OPT_MAC) || defined(OPT_BSD) || defined(OPT_UNIX)
#include <sys/socket.h>
#endif
typedef pthread_t thrid;
typedef void *thrarg;
typedef pthread_mutex_t	hmutex;
typedef pthread_cond_t	hsignal;
#define thrret void *
typedef bytex *pvirmemx;

#if defined(CPLUS_NEW_VER)
	typedef socklen_t socklenx;
#else
	typedef intx socklenx;
#endif

#endif //OPT_WIN

typedef void *hdynmdu;

typedef time_t time_x;
typedef struct tm xtm;

//파이프 언어에서 float 타입은 없다. 소수점 타임은 double로 통일, 이 기본타입 순서가 바뀌면(
//BYTEX_TP ~ DOUBLEX_TP까지) cmptype에 반영해야 한다. cmptype의 리턴을 받아서 이 타입간 차이로서 
//로직을 수행하기때문에 이 기본타입 구간에 추가해서도 않된다. 또한 mdb의 컬럼타입과 LONGX_TP, 
//DOUBLEX_TP, POINTX_TP의 인덱스가 일치해야한다.(mdb에서 DQI dml API에서 상수를 mdb로 전달할때 
//이 타입들을 사용하고 이 타입들은 바로 mdb의 COLUMN_T_LONG, COLUMN_T_DOUBLE, COLUMN_T_CHAR로 
//인식되어 사용되기때문)
#define NONE_TP			0	//타입이 없음
#define BYTEX_TP		1
#define SHORTX_TP		2
#define FLOATX_TP		3
#define INTX_TP			4
#define LONGX_TP		5	//mdb의 COLUMN_T_LONG과 일치해야함
#define DOUBLEX_TP		6	//mdb의 COLUMN_T_DOUBLE과 일치해야함
#define NUMBERX_TP		7
#define POINTX_TP		8	//기본타입이 아닌것의 대표타입, mdb의 COLUMN_T_CHAR과 일치해야함
#define POINT2X_TP		9
#define DATEX_TP		10	//날짜
#define DATETIMEX_TP	11	//날짜,시간
#define TIMEX_TP		12	//시간
#define ILOBX_TP		13	//레코드 슬롯내에 적재되는 블랍
#define BLOBX_TP		14	//여러 ManagedPage리스트로 구성되는 데이터
#define ELOBX_TP		15
#define REFERX_TP		16 //k107so.

#ifdef INTX64
#define INTX64_TP		LONGX_TP
#define ALIGN_INTX64(sz)	ALIGN_UNIT(sz)
#else 
#define INTX64_TP		INTX_TP
#define ALIGN_INTX64(sz)	ALIGN_INT(sz)
#endif

#define SHORTX_SZ	2
#define INTX_SZ		4
#define LONGX_SZ	8
#define UNIT_SZ		8
#define DUNIT_SZ	16
#define BYTEXP_SZ	256
#define SHORTEXP_SZ	65536

#define GIGA1		0X40000000

#define ORDNUM_LEN		16
#define ORDNUM_LEN2		32
#define NAME_LEN		512
#define PATH_LEN		1024

#define LEAD_LEN_FAST_BUF	18 //ORDNUM_LEN + 2
#define LEAD_LEN_FAST_BUF2	ORDNUM_LEN2

#define ALIGN_SIZE_(sz, align) ((sz/align) * align + align)
#define ALIGN_INT_(sz)	ALIGN_SIZE_(sz, sizeof(intx))
#define ALIGN_UNIT_(sz)	ALIGN_SIZE_(sz, sizeof(unit))

#define ALIGN_SIZE(sz, align) (sz % align ? ALIGN_SIZE_(sz, align) : sz)
#define ALIGN_INT(sz)	ALIGN_SIZE(sz, sizeof(intx))
#define ALIGN_UNIT(sz)	ALIGN_SIZE(sz, sizeof(unit))

#define ALIGN_DIV(i, div) (i % div ? i/div + 1 : i/div)
#define ALIGN_MUL(i, mul, div) ALIGN_DIV((i * mul), div)

#define ALIGN_SIZE2(sz, align) if(sz % align) sz = ALIGN_SIZE_(sz, align)
#define ALIGN_POINT(ptr, align) if((unit)ptr % align) ptr = (bytex *)ALIGN_SIZE_((unit)ptr, align)

#define ONLY_INT_ALIGN(sz) {\
	sz = ALIGN_INT(sz);\
	if(sz % sizeof(unit) == 0) sz += sizeof(intx);\
}

#define strleng	(intx)strlen

#ifdef __cplusplus
extern "C" {
#endif
// value는 반올림하고자 하는 실수값
// pos는 반올림하고자 하는 소수점 자리수
extern doublex roundx(doublex value, intx pos);
#ifdef __cplusplus
}
#endif

#endif










