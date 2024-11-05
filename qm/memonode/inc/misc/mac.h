
#include "misc/xt.h"

//csz - C컴파일러에서 계산된 브랍클래스의 크기에 블랍데이터의 크기를 더한것
//asz - csz 크기에 8바이트 정렬 패딩이 더해진 크기
#define LEN_SYS_CONST(syscon_t, dsz, csz, asz) {\
	csz = sizeof(syscon_t) + dsz;\
	asz = (intx)ALIGN_UNIT(csz);\
}
//lenPad - 블랍클래스 크기에 유닛사이즈 패딩 바이트를 더한크기
//lenBlob - 실제블랍데이터길이, lenPad + lenBlob하면 전채크기
#define SYS_CONST_SET(const_t, const_p, dsz, csz, asz) {\
	const_p->lenPad = (bytex)((asz - csz) + sizeof(const_t));\
	const_p->lenBlob = dsz;\
}
#define LEN_SYS_CONST2(dsz, csz, rsz, asz) {\
	rsz = csz + dsz;\
	asz = (intx)ALIGN_UNIT(rsz);\
}
#define SYS_CONST_SET2(const_p, dsz, csz, rsz, asz) {\
	const_p->lenPad = (asz - rsz) + csz;\
	const_p->lenBlob = dsz;\
}
#define CALC_LEN_SC(const_p) const_p->lenPad + const_p->lenBlob

#define HeadEnd(p) (p +1)
#define HeadBegin(p) (p -1)

#define ResetByte16(p) { *(unit *)p = 0; *((unit *)p + 1) = 0; }

#define SetStatusBit(status, ibit) status |= (1 << ibit)
#define ResetStatusBit(status, ibit) status &= ~(1 << ibit)
#define ScanStatusBit(status, ibit) (status & (1 << ibit))

#define CAST

#define GET_LOW4(l) ((~(unit)0 >> 32) & l )
//#define CHK_GET_LOW4(l) (sizeof(l) == sizeof(unit) ? GET_LOW4(l) : l)
#define CHK_GET_LOW4(l) l	//sysCurTime을 시작 시간과의 차이로 설정하는 것으로 바꿨기때문에 그대로 한다.

#define ASE	'&'
#define ANV	'^'