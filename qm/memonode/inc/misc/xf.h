#ifndef _H_XF
#define _H_XF

#include "misc/xt.h"
#include "misc/basys.h"

#ifdef OPT_WIN
#define xseek	_lseek
#define xsleep(t) Sleep(t*1000)
#define msleep(t) Sleep(t)
#define xlink(f) _unlink(f)
#define xcurrent_time()	(GetCurrentTime() / 1000)
#define mcurrent_time()	GetCurrentTime()
#define xgetpid	GetCurrentProcessId
#define xsend(_fd, _p, _n)	send(_fd, _p, _n, 0)
#define xrecv(_fd, _p, _n)	recv(_fd, _p, _n, 0)
#define xtime64(t) _time64(t)
#define xlocaltime64(t) _localtime64(t)
#define xmktime64(t) _mktime64(t)
#define PSFLONG "%s [%I64d] [%I64d]\n"
#define PSFLONG2 "%s [%I64d] %d\n"
#define DECL_CPU_USAGE(high1, kernel1_low, user1_low, idle1_low) uintx high1, kernel1_low, user1_low, idle1_low
#define xinitGetCpuUsage(high1, kernel1_low, user1_low, idle1_low) initGetCpuUsage(high1, kernel1_low, user1_low, idle1_low)
#define xgetCpuUsage(high1, kernel1_low, user1_low, idle1_low) getCpuUsage(high1, kernel1_low, user1_low, idle1_low)
#else

#define xseek	lseek
#define xsleep(t) sleep(t)
#define msleep(t) usleep(1000 *t)
#define xusleep(usec) usleep(usec)
#define xlink(f) unlink(f)
#define xcurrent_time()	time(nullx)
#define xgetpid	getpid
#define xsend(_fd, _p, _n)	write(_fd, _p, _n)
#define xrecv(_fd, _p, _n)	read(_fd, _p, _n)
#define PSFLONG "%s [%lld] [%lld]\n"
#define PSFLONG2 "%s [%lld] %d\n"
#define DECL_CPU_USAGE(high1, kernel1_low, user1_low, idle1_low)
#define xinitGetCpuUsage(high1, kernel1_low, user1_low, idle1_low)
#define xgetCpuUsage(high1, kernel1_low, user1_low, idle1_low) getCpuUsage()
#endif

#define BASE_YEAR	1900

#define xtime(t) time(t)
#define xlocaltime(t) localtime(t)
#define xmktime(t) mktime(t)

#define xgettime(when) {\
	time_x now;\
	xtime(&now);\
	when = *xlocaltime(&now);\
}
#define xtolower(str, tar) {\
	bytex *_p, *out = tar;\
	for(_p = str;*_p != '\0'; _p++, out++) {\
		if(*_p >= 'A' && *_p <= 'Z') {\
			*out = 'a' + (*_p - 'A');\
		} else *out = *_p;\
	}\
	*out = '\0';\
}
#define xtoupper(str, tar) {\
	bytex *_p, *out = tar;\
	for(_p = str;*_p != '\0'; _p++, out++) {\
		if(*_p >= 'a' && *_p <= 'z') {\
			*out = 'A' + (*_p - 'a');\
		} else *out = *_p;\
	}\
	*out = '\0';\
}
#define xinitcap(str) if(*str >= 'a' && *str <= 'z') *str = 'A' + (*str - 'a')

#ifdef __cplusplus
extern "C" {
#endif

extern thrid xthread_create(void *, void *);
extern void xthread_exit(void)	;
extern void xthread_free(thrid th);
extern intx xcompare_file_time(bytex *fname1, bytex *fname2);
extern hdynmdu xload_library(bytex *mdname);
extern hdynmdu xload_library2(bytex *mdname);
extern void xrelease_library(hdynmdu h);
extern void *xget_dynfunc(hdynmdu h, bytex *fname);
extern int xdelete_directory(bytex *path, intx recur);
extern intx getMacAddress(bytex *if_name, bytex *mac_addr);
extern uintx getMemInfo(unit &totmem);
extern unit xucurrenttime(void);
#ifdef OPT_WIN
extern void initGetCpuUsage(uintx &high1, uintx &kernel1_low, uintx &user1_low, uintx &idle1_low);
extern uintx getCpuUsage(uintx &high1, uintx &kernel1_low, uintx &user1_low, uintx &idle1_low);
extern void xgettimeofday(struct timeval *tv, struct timezonex *tz);
extern void xusleep(intx usec);
#elif OPT_LINUX
#define xgettimeofday(tv, tz) gettimeofday(tv, tz)
extern uintx getCpuUsage(void);
extern longx mcurrent_time();
#endif
#ifdef __cplusplus
}
#endif

#endif
