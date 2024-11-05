#ifndef _H_DTP
#define _H_DTP

#include "misc/xt.h"
#include "misc/xf.h"
#include "misc/expt.h"
#include "misc/cvt.h"

#ifdef __cplusplus
extern "C" {
#endif	
extern intx name2Month(bytex *name);
extern void XGetTime(xtm &when);
//ExecFormatChar2Date로 실행
//fmtcode - 생성되는 포맷코드(첫번째 바이트는 DATE_FMT, TIME_FMT, DTIME_FMT를 나타낸다.)
//return - 생성되는 포맷 코드 길이
extern void CodingChar2Date(bytex *fmtcode, bytex *formatm, intx fc_len);
extern intx Coding2Char(bytex *fmtcode, bytex *format, intx fc_len);

extern void printbsRight2Left(char *p, int nbyte);
extern void printbsLeft2Right(char *p, int nbyte);

#ifdef __cplusplus
}
#endif

#ifdef INTX64
#define COLUMN_T_INTX64		COLUMN_T_LONG
#else
#define COLUMN_T_INTX64		COLUMN_T_INT
#endif

#define COLUMN_T_NONE		0
#define COLUMN_T_BYTE		1
#define COLUMN_T_SHORT		2	//사실상 안쓴다.
#define COLUMN_T_FLOAT		3
#define COLUMN_T_INT		4	
#define COLUMN_T_LONG		5	//LONGX_TP와 인덱스가 일치해야 한다.
#define COLUMN_T_DOUBLE		6	//DOUBLEX_TP와 인덱스가 일치해야 한다.
#define COLUMN_T_NUMBER		7
#define COLUMN_T_CHAR		8	//POINTX_TP와 인덱스가 일치해야 한다.
#define COLUMN_T_VCHAR		9
#define COLUMN_T_DATE		10	//날짜
#define COLUMN_T_DATETIME	11	//날짜,시간
#define COLUMN_T_TIME		12	//시간
#define COLUMN_T_ILOB		13	//레코드 슬롯내에 적재되는 블랍
#define COLUMN_T_BLOB		14	//여러 ManagedPage리스트로 구성되는 데이터
#define COLUMN_T_ELOB		15	//데이터베이스 외부에 데이터 적재하는 타입

#define COLUMN_T_REFERENCE	16	//주소참조(seg_t_adr) 타입, 파이프언어에서 레코드의 참조 컬럼을
								//추적하면서 참조 레코드들을 해제(참조 레코드가 속하는 페이지의
								//aRecTable로 테이블을 획득하여 그 테이블에서 참조레코드 삭제를 수행)

#define IntUnder(t) t < COLUMN_T_LONG
#define DecimalUnder(t)	t < COLUMN_T_NUMBER 
#define NumberUnder(t) t < COLUMN_T_CHAR
#define CharUnder(t) t < COLUMN_T_DATE
#define DateUnder(t) t < COLUMN_T_ILOB
#define NumberRange(t) t == COLUMN_T_NUMBER
#define CharRange(t) t > COLUMN_T_NUMBER && t < COLUMN_T_DATE
#define DateRange(t) t > COLUMN_T_VCHAR && t < COLUMN_T_ILOB
#define BlobRange(t) t > COLUMN_T_TIME && t < COLUMN_T_REFERENCE
#define ExtTaggingBlob(t) t == COLUMN_T_BLOB || t == COLUMN_T_ELOB
#define NumberOver(t) t > COLUMN_T_DOUBLE
#define DateOver(t) t > COLUMN_T_VCHAR
#define NotCharRange(t) t < COLUMN_T_CHAR || t > COLUMN_T_VCHAR
#define NotExtTaggingBlob(t) t != COLUMN_T_BLOB && t != COLUMN_T_ELOB

//cvt.h의 decimal -> 스트링으로의 변환 매크로와 dtp.h의 넘버 -> 스트링으로의 변환 매크로만이 
//rsize에 널이 포함되지않고 다른 모든 매크로(데이트->스트링 변환과 sqllib.h와 같은 이외 파일)는 
//입출력 모두 널이 포함된 길이이다.

/********************************* free byte order macro *******************************/
/******************************** 네이티브 바이너리 매크로 *****************************/
#define BIT32_NEXP		8
#define BIT32_NMANTISA	23
#define BIT32_NRSTORMAN	24		//만티사의 히든비트를 확장한 자리수
#define BIT32_BIAS		127
#define BIT64_NEXP		11
#define BIT64_NMANTISA	52
#define BIT64_NRSTORMAN	53		//만티사의 히든비트를 확장한 자리수
#define BIT64_BIAS		1023
//sign부호 비트는 양수이면 0, 음수이면 1이다.
#define GetMantisaMask(bmsk, bits, lenMan, type) bmsk = ~((type)0) >> (bits - lenMan)
#define GetSign32(pd) *pd >> 31
#define GetUSign32(pd) (intx)(GetSign32(pd))
#define GetSign64(pd) *pd >> 63
#define GetUSign64(pd) (intx)(GetSign64(pd))
#define GetVSign32(d) *(uintx *)&d >> 31
#define GetVSign64(d) *(unit *)&d >> 63

#define MaskingMantisa(pd, bmsk, man) man = *pd & bmsk
#define GetExponent(pd, bmsk, lenMan, lenExp, bias, exp) {\
	exp = ((*pd & ~bmsk) >> lenMan) & ~(1 << lenExp);\
	exp -= bias;\
}
//mantisa의 히든비트를 살려낸다, 가수부에 hidden bit 한비트를 추가하고 exp를 하나 증가
#define AliveHidden(mantisa, lenMan, exp, type) {\
	mantisa |= ((type)1 << lenMan);\
	exp++;\
}
#define SetMantisa(pd, bmsk, man) *pd = bmsk & man
#define SetExponent(tp, pd, exp, lenMan, bias) {\
	exp += bias;\
	*pd = *pd | ((tp)exp << lenMan);\
}
#define SetSign(pd, sign, lenMan, lenExp) *pd = *pd | (sign << (lenMan + lenExp))
//지수는 가수가 0.1xxx에서 히든 처리하면 1.xxx되므로 하나감소
#define HiddenProcess exp--

#define TakeOutReal32(d, sign, exp, mantisa) {\
	uintx *pd = (uintx *)&d;\
	uintx b = 0;\
	GetMantisaMask(b, 32, BIT32_NMANTISA, uintx);\
	MaskingMantisa(pd, b, mantisa);\
	GetExponent(pd, b, BIT32_NMANTISA, BIT32_NEXP, BIT32_BIAS, exp);\
	sign = GetSign32(pd);\
	AliveHidden(mantisa, BIT32_NMANTISA, exp, intx);\
}
#define TakeInReal32(d, sign, exp, mantisa) {\
	uintx *pd = (uintx *)&d;\
	uintx b = 0;\
	GetMantisaMask(b, 32, BIT32_NMANTISA, uintx);\
	HiddenProcess;\
	SetMantisa(pd, b, mantisa);\
	SetExponent(uintx, pd, exp, BIT32_NMANTISA, BIT32_BIAS);\
	SetSign(pd, sign, BIT32_NMANTISA, BIT32_NEXP);\
}

#define TakeOutReal64(d, sign, exp, mantisa) {\
	unit *pd = (unit *)&d;\
	unit b = 0;\
	GetMantisaMask(b, 64, BIT64_NMANTISA, unit);\
	MaskingMantisa(pd, b, mantisa);\
	GetExponent(pd, b, BIT64_NMANTISA, BIT64_NEXP, BIT64_BIAS, exp);\
	sign = GetUSign64(pd);\
	AliveHidden(mantisa, BIT64_NMANTISA, exp, longx);\
}
#define TakeInReal64(d, sign, exp, mantisa) {\
	unit *pd = (unit *)&d;\
	unit b = 0;\
	GetMantisaMask(b, 64, BIT64_NMANTISA, unit);\
	HiddenProcess;\
	SetMantisa(pd, b, mantisa);\
	SetExponent(unit, pd, exp, BIT64_NMANTISA, BIT64_BIAS);\
	SetSign(pd, (longx)sign, BIT64_NMANTISA, BIT64_NEXP);\
}
#define BindReal(bunder, bover) bunder += bover
//bover - 소수점 상위를 적재는 실수, bunder - 소수점 하위를 적재는 실수
#define DevideMantisa32(dd, sign, exp, bover, bunder) {\
	intx mantisa;\
	TakeOutReal32(dd, sign, exp, mantisa);\
	if(sign == 1) dd *= -1;\
	if(exp >= BIT32_NRSTORMAN) {\
		bover = (intx)dd; bunder = 0.0;\
	} else if(exp >= 0) {\
		mantisa >>= (BIT32_NRSTORMAN - exp);\
		bover = mantisa;\
		bunder = dd - bover;\
	} else { bover = 0; bunder = dd; }\
}
/*void DevideMantisa32(floatx d, intx &sign, intx &exp, intx &bover, floatx &bunder)
{
	intx mantisa;
	floatx dd;
	TakeOutReal32(d, sign, exp, mantisa);
	dd = (sign == 1 ? d * -1 : d);
	if(exp >= BIT32_NRSTORMAN) {
		bover = (intx)dd; bunder = 0.0;
	} else if(exp >= 0) {
		mantisa >>= (BIT32_NRSTORMAN - exp);
		bover = mantisa;
		bunder = dd - bover;
	} else { bover = 0; bunder = dd; }
}*/
#define DevideMantisa64(dd, sign, exp, bover, bunder) {\
	longx mantisa;\
	TakeOutReal64(dd, sign, exp, mantisa);\
	if(sign == 1) dd *= -1;\
	if(exp >= BIT64_NRSTORMAN) {\
		bover = (longx)dd; bunder = 0.0;\
	} else if(exp >= 0) {\
		mantisa >>= (BIT64_NRSTORMAN - exp);\
		bover = mantisa;\
		bunder = dd - bover;\
	} else { bover = 0; bunder = dd; }\
}
/*void DevideMantisa64(doublex d, intx &sign, intx &exp, longx &bover, doublex &bunder)
{
	longx mantisa;
	doublex dd;
	TakeOutReal64(d, sign, exp, mantisa);
	dd = (sign == 1 ? d * -1 : d);
	if(exp >= BIT64_NRSTORMAN) {
		bover = (longx)dd; bunder = 0.0;
	} else if(exp >= 0) {
		mantisa >>= (BIT64_NRSTORMAN - exp);
		bover = mantisa;
		bunder = dd - bover;
	} else { bover = 0; bunder = dd; }
}*/
//size_field(1) + exp(1)+(prec/2)
#define GetNumberSize(prec) (2 + (prec % 2 ? prec / 2 + 1 : prec / 2))
#ifdef OPT_PURE32
#define LONG_ASCE_MAX 0x7fffffff		//오름차순 최대값
#define LONG_DESC_MIN 0x80000001		//내림차순 최소값
#else
#define LONG_ASCE_MAX 0x7fffffffffffffff		//오름차순 최대값
#define LONG_DESC_MIN 0x8000000000000001		//내림차순 최소값, 0x8000000000000000은 0
#endif
#define LONG_DESC_MAX -1						//내림차순 최대값
#define LONG_ASCE_MIN 1							//오름차순 최소값
#define LEN_LONG_LIMIT		19					//10진 표기법상 signed longx의 최대 길이 

#define NUM_2_INT_LIMIT		7
#define NUM_2_LONG_LIMIT	15
//******************************** 이하 넘버타입 매크로 *****************************
//컬럼선언이 넘버(p, s)에서 p가 7이하이면 floatx로 15이하이면 doublex로 저정한다. 이유는 floating
//point에서 각각 최대 9999999(2^23 about), 999999999999999(2^52 about)까지는 가수로서 직접저장되어 
//10진수나 100진수변환시 정확하게 변환될수있기 때문이다.
//아래 변환매크로에서 소수점이하 타겟 사이즈가 소스보다 더 적을경우 반올림을 해야하나 시스템에서 
//컬럼의 타입을 유닛 사이즈를 넘는 수는 넘버타입으로 하고 그 이하사이즈는 네이티브타입(각 타입별 
//계산의 최대값을 고려하여 오버플로우가 나지않는 타입으로 설정)으로 하기때문에 넘버타입에서 
//네이티브타으로는 변환이 없고 그 반대는 넘버타입의 크기가 더 클것이기때문에 반올림할경우는 없으므
//로 여기서 반올림 경우는 생략한다.
//값을 입력할때 컬럼의 넘버타입 정의를 기준으로 소수점 상위길이가 넘어가면 에러를 하위길이가 넘어가면
//반올림을 수행한후에 아래 매크로로 넘버로 변환한다.
//
//넘버데이터 정렬 포맷 - |size(1byte)|sign(1bit)|exp(7bit)|mantisa<---------- |

//precision - 5, scale - 2, ---> address진행 방향 <

//	|________|________|________|________|________|

//	^				  ^		   .				 ^
//	|				  |		   |				 |
// numhead			fpover	fpunder	 			numtail
//
//precision - 전체 넘버 길이, scale - 소수점 이하 길이, 그런데 여기서의 길이는 십진수가 아닌 
//백진수이므로 즉 두자리가 한바이트로 압축되므로 반환되는 rsize는 제시된 precision보다 준다.
//rsize - 실제 구성된 넘버 바이트 길이(size필드 포함된), numhead버퍼는 precision사이즈보다 한바이트
//(size필드가 추가되므로)더 커야한다.(비교를 정수단위로 할수없다. 음의 경우 부호비트(양은 1)를 포함하여 
//보수를 취하기때문 음은 양보다 클수없는데 정수비교하면 바이트오더가 틀리므로 이경우를 고려하여 일률적으로
//큰게 작은거 작은건 큰거와 같이 반대로 할수없기때문이다.)
//넘버를 write하고 나서 시작포인터는 fpover로 취하고 사이즈는 rsize가 된다.
//numhead를 사이즈 필드 다음 옵셋으로 증가시킨후 처리한다.
#define begNumWrite(numhead, numtail, fpover, fpunder, precision, scale, opow, upow, rsize) {\
	numhead++;\
	numtail = numhead + precision;\
	fpunder = numtail - scale;\
	fpover = fpunder - 1;\
	rsize = opow = upow = 0;\
}
#define IsOverHead(numhead, fpover) numhead >= fpover	//최선두 1바이트는 exp자리이므로 
#define IsUnderTail(numtail, fpunder) fpunder >= numtail
#define IsEON(p) *p == 0
//123456 195,13,35,57            C30D2339
//-123456 60,89,67,45            3C59432D
//소수점 상위 쓰기, sign - 1이면 양수 0이면 음수, 
//1을 더하여 쓴다, 음수이면 보수를 취하여 1을 더하여 쓴다.
//xx0000.0이면 소수점 이상 4개 0는 가수(맨티사, fpover)에 적재하지않고 opow만 증가한다.
#define WriteOverMantisa_(rsize, sign, opow, fpover, nb, bunder) {\
	if(nb == 0 && rsize == 0 && bunder == 0) opow++;\
	else {\
		if(sign == 0) nb = 100 - nb;\
		*fpover-- = (nb +1);\
		rsize++;\
		opow++;\
	}\
}
#define WriteOverMantisa(rsize, sign, opow, fpover, bover, bunder) {\
	sytex nb = (bytex)(bover % 100);\
	bover /= 100;\
	WriteOverMantisa_(rsize, sign, opow, fpover, nb, bunder);\
}
//소수점 하위 쓰기, 0.0000xx이면 소수점 이하 4개 0는 가수(맨티사, fpunder)에 적재하지않고 upow만 증가한다.
#define WriteUnderMantisa_(rsize, sign, upow, fpunder, nb, bover) {\
	if(nb == 0 && rsize == 0 && bover == 0) upow++;\
	else {\
		if(sign == 0) nb = 100 - nb;\
		*fpunder++ = (nb +1);\
		rsize++;\
		upow++;\
	}\
}
//bunder - floatx or doublex, 소수점이하, 0.~
#define WriteUnderMantisa(rsize, sign, upow, fpunder, bunder, bover) {\
	sytex nb;\
	bunder *= 100;\
	nb = (bytex)bunder;\
	bunder -= nb;\
	WriteUnderMantisa_(rsize, sign, upow, fpunder, nb, bover);\
}
//sign bit + exponet bit - 1 bytex, exp - -63 ~ 64, sign - 1이면 양수, 0이면 음수, 일단 부호비트를
//1로 설정하여 양수로 exp를 설정한후 음수이면 보수변환, 0xc0 - 64 bias + left most sign bit
#define WriteExponent(rsize, sign, exp, fpover) {\
	*fpover = exp + 0xc0;\
	if(sign == 0) *fpover = ~*fpover;\
	rsize++;\
}
//맨끝을 0(EON-end of number)로 마감하는데 exp바이트를 포함하여 멘티사의 길이가 4의 배수로 
//떨어지지않으면 그 나머지를 0로 채우고 맨끝에 0(EON)을 하나더 write한다. - 비교연산할때 intx단위
//로 비교연산하기위해(끝은 EON으로 체크)
//#define endNumWrite(rsize, fpunder) {\
//	intx rest = (4 - rsize % 4) + 1;\
//	intx i;\
//	if(rest == 5) rest = 1;\
//	for(i = 0; i < rest; i++) *fpunder++ = (bytex)0;\
//} - depricate - intx로 비교하면 역워드시스템은 비교가 반대로 되므로 한바이트씩 비교한다.

//주어진 fpover는 변환된 넘버데이터의 선두인 (sign + exp)바이트로서 하나 더 감소하여 맨 선두에 사이즈 필드
//가 한 바이트 추가된 rsize길이를 설정한다. 0이 입력되면 여기서 1을 넣어준다.
#define endNumWrite(rsize, sign, opow, upow, fpover, fpunder) {\
	intx exp = (opow ? opow : upow * -1);\
	if(exp == 0) {\
		*fpover-- = 1;\
		rsize++;\
		exp++;\
	}\
	WriteExponent(rsize, sign, exp, fpover);\
	*(--fpover) = ++rsize;\
}
#define SizeOfNumber(pnum) *pnum
//(opt, utp, bit)는 (intx, floatx, 32) or (longx, doublex, 64)이다. sign은 ieee와 넘버표현이
//서로 반대이므로 ieee로 읽은 부호를 반대로 변환한다. 
//fpover로 넘버로 쓰여진 시작포인터(맨 선두의 사이즈 필드를 가리키는 포인터)를 리턴받는다.(numtar공간내의)
//rsize - 맨 선두의 사이즈 필드 1바이트가 포함된 변환된 넘버데이터 길이
//exp - 일률적으로 intx타입으로 해도되나 경고에러를 피하기위해 otp데이터 타입과 일치시킨다.(혹시 
//		나중에 에러나면 intx로 되돌림)
#define Real2Number(otp, utp, bit, real, numtar, precision, scale, rsize, fpover) {\
	intx sign, opow, upow;\
	bytex *fpunder, *numhead = numtar, *numtail;\
	otp bover, exp;\
	utp bunder, dd = real;\
	\
	DevideMantisa##bit(dd, sign, exp, bover, bunder);\
	sign = (sign ? 0 : 1);\
	begNumWrite(numhead, numtail, fpover, fpunder, precision, scale, opow, upow, rsize);\
	for(;bover && !(IsOverHead(numhead, fpover));) \
		WriteOverMantisa(rsize, sign, opow, fpover, bover, bunder);\
	for(;bunder && !(IsUnderTail(numtail, fpunder));) \
		WriteUnderMantisa(rsize, sign, upow, fpunder, bunder, bover);\
	endNumWrite(rsize, sign, opow, upow, fpover, fpunder);\
}
/*void Float2Number(floatx real, bytex *numhead, intx precision, intx scale, intx &rsize, bytex *&fpover)
{
	intx sign, exp, opow, upow;
	bytex *fpunder, *numtail;
	intx bover;
	floatx bunder;
	
	DevideMantisa32(real, sign, exp, bover, bunder);
	sign = (sign ? 0 : 1);
	begNumWrite(numhead, numtail, fpover, fpunder, precision, scale, opow, upow, rsize);
	for(;bover && !(IsOverHead(numhead, fpover));) 
		WriteOverMantisa(rsize, sign, opow, fpover, bover, bunder);
	for(;bunder && !(IsUnderTail(numtail, fpunder));) 
		WriteUnderMantisa(rsize, sign, upow, fpunder, bunder, bover);
	endNumWrite(numhead, rsize, sign, opow, upow, fpover, fpunder);
}*/
#define Float2Number(real, num, prec, scale, rsize, out) \
    Real2Number(intx, floatx, 32, real, num, prec, scale, rsize, out)
#define Double2Number(real, num, prec, scale, rsize, out) \
	Real2Number(longx, doublex, 64, real, num, prec, scale, rsize, out)
//under는 신경않써도 됨, natual의 타입은 tp와 같아야한다. fpunder는 끝을 가리킴
//numtar - 출력버퍼, fpover로 넘버로 변환된 데이터의 시작포인터(맨 선두의 사이즈 필드를 가리키는 포인터)
//를 리턴받는다.(numtar공간내의), rsize - 맨 선두의 사이즈 필드 1바이트가 포함된 변환된 넘버데이터 길이
#define Integer2Number(tp, bit, natual, numtar, precision, rsize, fpover) {\
	intx sign, opow, upow;\
	bytex *fpunder, *numhead = numtar, *numtail;\
	tp bover;\
	doublex bunder = 0;\
	\
	if(GetVSign##bit(natual)) { sign = 0; bover = natual * -1; }\
	else { sign = 1; bover = natual; }\
	begNumWrite(numhead, numtail, fpover, fpunder, precision, 0, opow, upow, rsize);\
	for(;bover && !(IsOverHead(numhead, fpover));) \
		WriteOverMantisa(rsize, sign, opow, fpover, bover, bunder);\
	endNumWrite(rsize, sign, opow, upow, fpover, fpunder);\
}
/*void Int2Number(intx natual, bytex *numhead, intx precision, intx &rsize, bytex *&fpover)
{
	intx sign, exp, opow, upow;
	bytex *fpunder, *numtail;
	intx bover = natual;
	doublex bunder;
	
	sign = (GetVSign32(natual) ? 0 : 1);
	begNumWrite(numhead, numtail, fpover, fpunder, precision, 0, opow, upow, rsize);
	for(;bover && !(IsOverHead(numhead, fpover));) 
		WriteOverMantisa(rsize, sign, opow, fpover, bover, bunder);
	endNumWrite(numhead, rsize, sign, opow, upow, fpover, fpunder);
}*/
#define Int2Number(nat, num, prec, rsize, out) Integer2Number(intx, 32, nat, num, prec, rsize, out)
#define Long2Number(nat, num, prec, rsize, out) Integer2Number(longx, 64, nat, num, prec, rsize, out)
//numstr입력 넘버스트링, numsor - 출력 넘버 버퍼, 
//fpover로 변환된 넘버데이터의 시작 포인터(맨 선두의 사이즈 필드를 가리키는 포인터)를 리턴받는다.(numtar
//공간내의), rsize - 맨 선두의 사이즈 필드 1바이트가 포함되어 변환된 전체 넘버데이터 길이
#define Char2Number(numstr, numtar, precision, scale, rsize, fpover) {\
	intx sign, opow, upow;\
	bytex *fpunder, *numhead = numtar, *numtail, *op, *up, *p, *beg, nb;\
	\
	begNumWrite(numhead, numtail, fpover, fpunder, precision, scale, opow, upow, rsize);\
	for(op = numstr; (*op == ' ') || (*op == '\n') || (*op == '\t') || (*op == '\r'); op++);\
	sign = (*op == '-' ? 0 : 1);\
	if(*op == '+' || *op == '-') op++;\
	for(;*op == '0'; op++);\
	beg = op;\
	for(;*op != '.' && *op != '\0'; op++);\
	if(*op == '.') up = op + 1;\
	else up = nullx;\
	if(--op < beg) op = nullx;\
	if(op) {\
		for(;!(IsOverHead(numhead, fpover));) {\
			if(op >= beg && (*op >= '0') && (*op <= '9')) {nb = *op - '0'; op--; }\
			else break;\
			if(op >= beg && ((*op >= '0') && (*op <= '9'))) { nb += ((*op - '0') * 10); op--; }\
			WriteOverMantisa_(rsize, sign, opow, fpover, nb, up);\
		}\
	}\
	if(up) {\
		for(p = up; *p != '\0'; p++);\
		for(p--; up < p && *p == '0'; p--);\
		if(p == up && *p == '0') up = nullx;\
		else *++p = '\0';\
		for(;*up != '\0' && !(IsUnderTail(numtail, fpunder));) {\
			if((*up >= '0') && (*up <= '9')) { nb = (*up - '0') * 10; up++; }\
			else break;\
			if((*up >= '0') && (*up <= '9')) { nb += (*up - '0'); up++; }\
			WriteUnderMantisa_(rsize, sign, upow, fpunder, nb, op);\
		}\
	}\
	endNumWrite(rsize, sign, opow, upow, fpover, fpunder);\
}


#define ReadExponent(numhead, sign, exp) {\
	sign = *numhead & 0x80;\
	exp = (sign ? *numhead & 0x7f : ~*numhead & 0x7f) - 64;\
}
//fpover의 끝 체크는 IsOverHead로 하고 fpunder의 끝체크는 IsUnderTail로 한다.
//그리고 초기상태에서 *fpunder의 값이 끝이면 소수점이하는 없는 것이다.
//numhead - 넘버데이터의 맨선두(사이즈필드) 포인터
//rsize - 안에서 구해진 실제 넘버데이터 길이(사이즈필드 길이가 포함된), 
//맨티사의 길이는 rsize - 2byte[size + (sign+exp)]
//pot - exp의 절대값이 실제 가수(멘티사)보다 클경우 그 차이만큼 100진수 곱(나누기)을 해야한다.
#define begNumRead_(sign, exp, numhead, numtail, rsize, fpover, fpunder, pot) {\
	intx nmantisa;\
	rsize = (*numhead++ - 1);\
	numtail = numhead + rsize;\
	nmantisa = rsize -1;\
	pot = 0;\
	ReadExponent(numhead, sign, exp);\
	if(exp > nmantisa) {\
		pot = exp - nmantisa;\
		fpover = numhead + nmantisa;\
	} else if(exp >= 0) fpover = numhead + exp;\
	else {\
		if(exp * -1 > nmantisa) pot = exp * -1 - nmantisa;\
		fpover = numhead;\
	}\
	fpunder = fpover +1;\
}
#define begNumRead(utp, sign, exp, numhead, numtail, rsize, fpover, fpunder, bover, bunder, opow, upow, pot) {\
	begNumRead_(sign, exp, numhead, numtail, rsize, fpover, fpunder, pot);\
	bover = 0;\
	bunder = (utp)0.0;\
	if(sign) { opow = 1; upow = (utp)0.01; }\
	else { opow = -1; upow = (utp)-0.01; }\
}

#define ReadOverMantisa_(fpover, sign, nb) {\
	nb = *fpover-- - 1;\
	if(sign == 0) nb = 100 - nb;\
}
//bover - 초기값은 0, opow - 초기값은 1에서 계속 100씩 곱해지는 수
#define ReadOverMantisa(fpover, sign, bover, opow) {\
	ubytex nb;\
	ReadOverMantisa_(fpover, sign, nb);\
	bover += (nb * opow);\
	opow *= 100;\
}
#define ReadUnderMantisa_(fpunder, sign, nb) {\
	nb = *fpunder++ - 1;\
	if(sign == 0) nb = 100 - nb;\
}
//utp - upow타입, bunder - 초기값은 0.0, upow - 초기값은 sign이 음수이면(0이면) -0.01, 
//양수이면 0.01에서 계속 0.01(1/100)씩 곱해지는(나뉘어지는) 수
#define ReadUnderMantisa(utp, fpunder, sign, bunder, upow) {\
	ubytex nb;\
	ReadUnderMantisa_(fpunder, sign, nb);\
	bunder += (nb * upow);\
	upow *= (utp)0.01;\
}
//만약 나중에 제대로 동작안한다면 아래 커맨트 부분을 살리고 그 바로 위에것을 삭제해볼것 - 사실 똑같지만
//output은 bunder이고 이것의 타입은 utp와 같아야한다. 
//numsor - 넘버데이터의 맨선두(사이즈필드) 포인터
//rsize - 안에서 설정되는 실제 넘버데이터(numsor)의 길이(맨선두 사이즈 한바이트 길이가 포함된)
#define Number2Real(otp, utp, numsor, rsize, bunder) {\
	intx sign, exp, pot, i;\
	bytex *numhead = numsor, *numtail, *fpover, *fpunder;\
	otp bover, opow;\
	utp upow;\
	begNumRead(utp, sign, exp, numhead, numtail, rsize, fpover, fpunder, bover, bunder, opow, upow, pot);\
	for(;!(IsOverHead(numhead, fpover));) ReadOverMantisa(fpover, sign, bover, opow);\
	for(;!(IsUnderTail(numtail, fpunder));) ReadUnderMantisa(utp, fpunder, sign, bunder, upow);\
	if(pot) {\
		if(exp >= 0) {\
			for(i = 0;i < pot; i++) bover *= 100;\
			bunder = 0;\
			BindReal(bunder, bover);\
			/*bunder = bover;*/\
		} else {\
			for(i = 0;i < pot; i++) bunder *= (utp)0.01;\
		}\
	} else BindReal(bunder, bover);\
}
//bunder - 출력변수이고 타입은 floatx
#define Number2Float(num, rsize, bunder) Number2Real(intx, floatx, num, rsize, bunder) 
//bunder - 출력변수이고 타입은 doublex
#define Number2Double(num, rsize, bunder) Number2Real(longx, doublex, num, rsize, bunder)
//numsor - 넘버데이터의 맨선두(사이즈필드) 포인터
//bover - 출력변수이고 타입은 tp와 같아야한다. bunder는 신경않써도 됨
#define Number2Integer(tp, numsor, rsize, bover) {\
	intx sign, exp, pot;\
	bytex *numhead = numsor, *numtail, *fpover, *fpunder;\
	tp opow;\
	unifltx upow, bunder;\
	begNumRead(unifltx, sign, exp, numhead, numtail, rsize, fpover, fpunder, bover, bunder, opow, upow, pot);\
	for(;!(IsOverHead(numhead, fpover));) ReadOverMantisa(fpover, sign, bover, opow);\
}
/*void Number2Int(bytex *numhead, intx rsize, intx bover)
{
	intx sign, exp, pot;
	bytex *fpover, *fpunder;
	intx opow;
	longx bunder;
	doublex upow;
	begNumRead(sign, exp, numhead, rsize, fpover, fpunder, bover, bunder, opow, upow, pot);
	for(;!(IsOverHead(numhead, fpover));) ReadOverMantisa(fpover, sign, bover, opow);
}*/
#define Number2Int(num, rsize, bover) Number2Integer(intx, num, rsize, bover)
#define Number2Long(num, rsize, bover) Number2Integer(longx, num, rsize, bover)

#define NUM2STR_LEN		296 //digitFmt2Char로 코딩된 넘버포맷으로 넘버를 변환할때의 최대 길이(코딩포맷을 E 지수
						//포맷으로 변경하면 이 길이도 변경해야함)
#define MAX_PRECISION	76
#define MAX_SCALE		38
#define MAX_INTEGER		38	//소수점 상위 최대 길이
//numsor - 넘버데이터의 맨선두(사이즈필드) 포인터
//si - 출력 문자버퍼의 시작포인터이고 사이즈는 1(부호) + MAX_SCALE(최대 정밀도) + 
//128(64 *2[exp가 100진수이므로 2자리]) * 2(소수점 상하위)이므로 295이다. 
//pover는 출력 변환된 숫자문자열의 시작포인터이고 안에서 일단 중간으로 이동시키는데 버퍼사이즈의 
//반인 148 옵셋 뒤로 이동한다.
//rsize - pover로부터 변환된 넘버스트링 끝까지의 길이(널제외),
#define Number2Char(numsor, si, rsize, pover) {\
	intx sign, exp, pot, i, underit;\
	bytex *numhead = numsor, *numtail, *fpover, *fpunder, *punder, nb;\
	\
	pover = si + 148;\
	punder = pover +1;\
	begNumRead_(sign, exp, numhead, numtail, rsize, fpover, fpunder, pot);\
	if(!(IsUnderTail(numtail, fpunder))) *punder++ = '.';\
	if(pot) {\
		if(exp >= 0) {\
			for(i = 0;i < pot; i++) { *pover-- = '0'; *pover-- = '0'; }\
		} else {\
			for(i = 0;i < pot; i++) { *punder++ = '0'; *punder++ = '0'; }\
		}\
	}\
	for(;!(IsOverHead(numhead, fpover));) {\
		ReadOverMantisa_(fpover, sign, nb);\
		*pover-- = '0' + nb % 10;\
		*pover-- = '0' + nb / 10;\
	}\
	if(*++pover == '0') pover++;\
	if(*pover == '.') *--pover = '0';\
	if(sign == 0) *--pover = '-';\
	if(IsEON(fpunder)) underit = 0;\
	else underit = 1;\
	for(;!(IsUnderTail(numtail, fpunder));) {\
		ReadUnderMantisa_(fpunder, sign, nb);\
		*punder++ = '0' + nb / 10;\
		*punder++ = '0' + nb % 10;\
	}\
	/*k473so.if(underit && *(punder -1) == '0') *--punder = '\0';\
	else*/ *punder = '\0';\
	rsize = (intx)(punder - pover);\
}
//numsor - 넘버데이터의 맨선두(사이즈필드) 포인터
//si - 출력 문자버퍼의 시작포인터이고 사이즈는 1(부호) + MAX_SCALE(최대 정밀도) + 5('E+000')+ 이므로 44이다.
//pover는 출력 변환된 숫자문자열의 시작포인터이고 안에서 일단 중간으로 39옵셋 만큼 뒤로 이동한다.
//넘버 exp정보는 백진수이므로 2를 곱하여 1을 감소시킨다.(0.xxxx~로 설정된것을 x.xxx~로 표현하므로)
#define Number2IEChar(numsor, si, rsize, pover, exp) {\
	intx sign, pot, overit;\
	bytex *numhead = numsor, *numtail, *fpover, *fpunder, *punder, nb;\
	\
	pover = si + 39;\
	punder = pover +1;\
	begNumRead_(sign, exp, numhead, numtail, rsize, fpover, fpunder, pot);\
	if(exp != 0) exp = exp * 2 -1;\
	if(!(IsOverHead(numhead, fpover))) overit = 1;\
	else overit = 0;\
	for(;!(IsOverHead(numhead, fpover));) {\
		ReadOverMantisa_(fpover, sign, nb);\
		*pover-- = '0' + nb % 10;\
		*pover-- = '0' + nb / 10;\
	}\
	pover++;\
	for(;!(IsUnderTail(numtail, fpunder));) {\
		exp += 2;\
		ReadUnderMantisa_(fpunder, sign, nb);\
		*punder++ = '0' + nb / 10;\
		*punder++ = '0' + nb % 10;\
	}\
	if(*pover == '0') { pover++; exp--; }\
	*(pover -1) = *pover;\
	*pover = '.';\
	pover--;\
	if(sign == 0) *--pover = '-';\
	if(*(punder -1) == '0') punder--;\
	if(*(punder -1) == '.') *punder++ = '0';\
	*punder = '\0';\
	rsize = (intx)(punder - pover);\
}
/*void Number2IEChar(bytex *numhead, intx rsize, bytex *&pover, intx &exp)
{
	intx sign, pot, i, overit;
	bytex *fpover, *fpunder, *punder, nb;
	
	pover += 44;
	punder = pover +1;
	begNumRead_(sign, exp, numhead, rsize, fpover, fpunder, pot);
	if(exp != 0) exp = exp * 2 -1;
	if(!(IsOverHead(numhead, fpover))) overit = 1;
	else overit = 0;
	for(;!(IsOverHead(numhead, fpover));) {
		ReadOverMantisa_(fpover, sign, nb);
		*pover-- = '0' + nb % 10;
		*pover-- = '0' + nb / 10;
	}
	pover++;
	for(;!(IsEON(fpunder));) {
		exp += 2;
		ReadUnderMantisa_(fpunder, sign, nb);
		*punder++ = '0' + nb / 10;
		*punder++ = '0' + nb % 10;
	}
	if(*pover == '0') { pover++; exp--; }
	*(pover -1) = *pover;
	*pover = '.';
	pover--;
	if(sign == 0) *--pover = '-';
	if(*(punder -1) == '0') {
		punder--;
		//if(overit == 0) exp++;
	}
	if(*(punder -1) == '.') *punder++ = '0';
	*punder = '\0';
}*/

/******************************************* char to date ******************************/
#ifdef DEF_SRC_DTP
bytex *monthName[] = {"JANUARY  ", "FEBRUARY ", "MARCH    ", "APRIL    ", "MAY      ", 
	"JUNE     ", "JULY     ", "AUGUST   ", "SEPTEMBER", "ACTOBER  ", "NOVEMBER ", "DECEMBER "};
bytex *monthName2[] = {"january", "february", "march", "april", "may", "june", "july", 
	"august", "september", "actober", "november", "december"};
bytex *dayName[] = {"일요일", "월요일", "화요일", "수요일", "목요일", "금요일", "토요일"};
#else
extern bytex *monthName[];
extern bytex *monthName2[];
extern bytex *dayName[];
#endif

#define SetYearMDay(t_date, _tar, year, month, day) {\
	if(t_date == COLUMN_T_DATE) {\
		*(uintx *)_tar = 0;\
		SetDateByte(*(uintx *)_tar, year, (YY_POS - 4));\
		SetDateByte(*(uintx *)_tar, month, (MM_POS - 4));\
		SetDateByte(*(uintx *)_tar, day, (DD_POS - 4));\
	} else {\
		ResetDateByte2(_tar);\
		SetDateByte(*(unit *)_tar, (unit)year, YY_POS);\
		SetDateByte(*(unit *)_tar, (unit)month, MM_POS);\
		SetDateByte(*(unit *)_tar, (unit)day, DD_POS);\
	}\
}	
#define GetYearMDay(_imem, t_date, _sor, _tar, year, month, day) {\
	if(t_date == COLUMN_T_DATE) {\
		IMemAlloc(bytex, _imem, sizeof(intx), _tar);\
		year = (intx)VGetDateByte(*(uintx *)_sor, (YY_POS - 4), 16);\
		month = (intx)VGetDateByte(*(uintx *)_sor, (MM_POS - 4), 8);\
		day = (intx)VGetDateByte(*(uintx *)_sor, (DD_POS - 4), 8);\
	} else {\
		IMemAlloc(bytex, _imem, sizeof(unit), _tar);\
		*(unit *)_tar = *(unit *)_sor;\
		year = (intx)VGetDateByte(*(unit *)_sor, YY_POS, 16);\
		month = (intx)VGetDateByte(*(unit *)_sor, MM_POS, 8);\
		day = (intx)VGetDateByte(*(unit *)_sor, DD_POS, 8);\
	}\
}

#define SetStruct2Date(sor, tar) {\
	uintx val;\
	val = sor.tm_year + BASE_YEAR;\
	SetDateByte(tar, val, (YY_POS - 4));\
	val = sor.tm_mon + 1;\
	SetDateByte(tar, val, (MM_POS - 4));\
	val = sor.tm_mday;\
	SetDateByte(tar, val, (DD_POS - 4));\
}
#define SetStruct2DTime(sor, tar) {\
	unit val;\
	val = sor.tm_year + BASE_YEAR;\
	SetDateByte(tar, val, YY_POS);\
	val = sor.tm_mon + 1;\
	SetDateByte(tar, val, MM_POS);\
	val = sor.tm_mday;\
	SetDateByte(tar, val, DD_POS);\
	val = sor.tm_hour;\
	SetDateByte(tar, val, HH_POS);\
	val = sor.tm_min;\
	SetDateByte(tar, val, MI_POS);\
	val = sor.tm_sec;\
	SetDateByte(tar, val, SS_POS);\
}
#define SetStruct2Time(sor, tar) {\
	uintx val;\
	val = sor.tm_hour;\
	SetDateByte(tar, val, HH_POS);\
	val = sor.tm_min;\
	SetDateByte(tar, val, MI_POS);\
	val = sor.tm_sec;\
	SetDateByte(tar, val, SS_POS);\
}
#define CvtLong2Date(when, set_date_mac, _sor, _tar) {\
	time_x tt = (time_x)*(longx *)_sor;\
	xtm *p = xlocaltime(&tt);\
	if(p == nullx) throwExpt(-1, "CvtLong2Date localtime fail\n");\
	when = *p;\
	_tar = 0;\
	set_date_mac(when, _tar);\
}
//sor - yyyymmdd, tar - xtm
#define SetDateStr2Struct(sor, tar) {\
	sytex c;\
	c = *(sor +4);\
	*(sor +4) = '\0';\
	tar.tm_year = atoi(sor) - BASE_YEAR;\
	*(sor +4) = c;\
	\
	c = *(sor +6);\
	*(sor +6) = '\0';\
	tar.tm_mon = atoi(sor +4) - 1;\
	*(sor +6) = c;\
	\
	c = *(sor +8);\
	*(sor +8) = '\0';\
	tar.tm_mday = atoi(sor +6);\
	*(sor +8) = c;\
	tar.tm_hour = 0;\
	tar.tm_min = 0;\
	tar.tm_sec = 0;\
}
#define SetDate2Struct(sor, tar) {\
	tar.tm_year = (intx)VGetDateByte(*(uintx *)sor, (YY_POS - 4), 16) - BASE_YEAR;\
	tar.tm_mon = (intx)VGetDateByte(*(uintx *)sor, (MM_POS - 4), 8) -1;\
	tar.tm_mday = (intx)VGetDateByte(*(uintx *)sor, (DD_POS - 4), 8);\
	tar.tm_hour = 0;\
	tar.tm_min = 0;\
	tar.tm_sec = 0;\
}
#define SetDTime2Struct(sor, tar) {\
	tar.tm_year = (intx)VGetDateByte(*(unit *)sor, YY_POS, 16) - BASE_YEAR;\
	tar.tm_mon = (intx)VGetDateByte(*(unit *)sor, MM_POS, 8) -1;\
	tar.tm_mday = (intx)VGetDateByte(*(unit *)sor, DD_POS, 8);\
	tar.tm_hour = (intx)VGetDateByte(*(unit *)sor, HH_POS, 8);\
	tar.tm_min = (intx)VGetDateByte(*(unit *)sor, MI_POS, 8);\
	tar.tm_sec = (intx)VGetDateByte(*(unit *)sor, SS_POS, 8);\
}
#define SetTime2Struct(sor, tar) {\
	tar.tm_year = BASE_YEAR;\
	tar.tm_mon = 0;\
	tar.tm_mday = 1;\
	tar.tm_hour = (intx)VGetDateByte(*(uintx *)sor, HH_POS, 8);\
	tar.tm_min = (intx)VGetDateByte(*(uintx *)sor, MI_POS, 8);\
	tar.tm_sec = (intx)VGetDateByte(*(uintx *)sor, SS_POS, 8);\
}

#define CvtDate2Long(when, set_struct_mac, _sor, _tar) {\
	set_struct_mac(_sor, when);\
	*(longx *)_tar = (longx)xmktime(&when);\
}
#define CopyCvtDate2Long(imem, when, set_struct_mac, _sor, _tar) {\
	IMemAlloc(bytex, imem, sizeof(longx), _tar);\
	CvtDate2Long(when, set_struct_mac, _sor, _tar);\
}

#define GetTime2Long(tar) {\
	xtm when;\
	XGetTime(when);\
	tar = (longx)xmktime(&when);\
}
#define GetSysTime(date) {\
	xtm when;\
	XGetTime(when);\
	SetStruct2DTime(when, date);\
}
#define SetSysTimeSeq(date, seq) {/*k361so.date 8바이트의 마지막 1바이트에 seq를 설정한다. seq는 unit type에 1바이트 시퀀스 값이 적재되서 호출되야함*/\
	GetSysTime(date);\
	SetDateByte(date, seq, 0);\
}
#define GetCurYear(year) {\
	xtm when;\
	XGetTime(when);\
	year = when.tm_year + BASE_YEAR;\
}
#define GetCurHour(hour) {\
	xtm when;\
	XGetTime(when);\
	hour = when.tm_hour;\
}
#define GetCurWDay(year, month, day, wday) {\
	intx sum = 0, i;\
	wday = (year-1)*365 + (year-1)/4 - (year-1)/100 + (year-1)/400;\
	for(i = 1;i < month; i++) {\
		switch(i) {\
			case 4: case 6: case 9: case 11:\
				wday +=30; break;\
			case 2:\
				if(year % 400 == 0 || year % 4 == 0 && year % 100 != 0) wday += 29;\
				else wday += 28;\
				break;\
            default:\
				wday += 31; break;\
		}\
	}\
	wday += day;\
	wday %= 7;\
}
#define GetLastDay(year, month, day) {\
	if(((month <= 7 && month%2==1) || (month>=8 && month%2==0)) && month != 2) day = 31;\
	else if(month == 2) {\
		if(year%4 == 0) day = 29;\
		else day = 28;\
	} else day = 30;\
}
//date에 우측으로부터 idx바이트 번째에 val값을 삽입한다.
#define SetDateByte(date, val, idx) date |= (val << (idx * 8))
//date에서 우측으로부터 idx바이트 번째의 nbit사이즈 값을 가져온다.
#define GetDateByte(date, idx, nbit) (date >> (idx * 8)) & ~(-1 << nbit)
//date에 우측으로부터 idx바이트 번째의 값을 리셋한다.
#define VGetDateByte(date, idx, nbit) (GetDateByte(date, idx, nbit))
#define ResetDateByte(date, val, idx) {\
	val = 0xff;\
	date &= ~(val << (idx * 8));\
}
//datetime형의 상위4바이트 날짜부분을 리셋한다.
#define ResetDateByte2(_date) *(unit *)_date &= (((unit)-1) >> 32)

#define DTCD_YY2	0
#define DTCD_YY4	1
#define DTCD_MM		2
#define DTCD_DD		3
#define DTCD_MON	4
#define DTCD_DM		5
#define DTCD_H12	6
#define DTCD_H24	7
#define DTCD_MI		8
#define DTCD_SS		9
//date type의 각 항목별로 값이 주어지지않으면 디폴트값을 설정하는 명령어
#define DTCD_S_YR		10
#define DTCD_S_MM		11
#define DTCD_S_DD		12
#define DTCD_S_H24		13
#define DTCD_S_MI		14
#define DTCD_S_SS		15
#define DTCD_END		16
#define DTCD_AM			17
//date item의 각 항목의 위치
#define YY_POS	6
#define MM_POS	5
#define DD_POS	4
#define HH_POS	3
#define MI_POS	2
#define SS_POS	1

#define COL_DATE_LEN	4
#define COL_DTIME_LEN	8
#define COL_TIME_LEN	4

#define DATE_FMT		2
#define TIME_FMT		1
#define DTIME_FMT		3

#define MARK_DATE_FMT(fmt_head) fmt_head |= DATE_FMT
#define MARK_TIME_FMT(fmt_head) fmt_head |= TIME_FMT

#define IS_DATE_FMT(fmt_head)	fmt_head == DATE_FMT
#define IS_TIME_FMT(fmt_head)	fmt_head == TIME_FMT
#define IS_DTIME_FMT(fmt_head)	fmt_head == DTIME_FMT

#include "misc/nls.h"
//date타입은 각각 COLUMN_T_DATETIME, COLUMN_T_DATE, COLUMN_T_TIME이고 그 포맷은
//yyyymmddhhmissxx(8) or yyyymmdd(4) or hhmissxx(4)이다(xx는 의미없음)
//base는 각각 0, -4, 0이고 a_t는 각각 unit, uintx, uintx이다.
template <class a_t> void ExecFormatChar2Date_(void *hour_fac, bytex *fmtcode, bytex *strdt, a_t &date, intx base) 
{
	bytex *pc = fmtcode +1;//첫번째 바이트는 데이터 세부 타입을 나타낸다.
	bytex *pd = strdt;
	bytex *p;
	a_t val, save_val = 60;
	intx len, pos, rv;
	ushortx year;
	sytex c;

	date = (a_t)0;
	for(len = 0;*pc != DTCD_END;) {
		val = 0;
		switch(*pc) {
		case DTCD_YY2:
			len = 2;
			LOSStn2Integer(pd, val, len);
			if(len < 0) break;
			pc++;//세기를 획득하기위해 
			memcpy(&year, pc, 2);
			val += year;//년도에 세기를 더한다.
			if(val < 1 || val > 65535) throwExpt(-1, "ExecFormatChar2Date_ out range year\n");
			pc += 2;
			SetDateByte(date, val, (YY_POS + base));			
			continue;
		case DTCD_YY4:
			len = 4; 
			LOSStn2Integer(pd, val, len);
			if(len < 0) break;
			if(val < 1 || val > 65535) throwExpt(-1, "ExecFormatChar2Date_ out range year\n");
			SetDateByte(date, val, (YY_POS + base));
			break;
		case DTCD_MM:
			len = 2; 
			LOSStn2Integer(pd, val, len);
			if(len < 0) break;
			if(val < 1 || val > 12) throwExpt(-1, "ExecFormatChar2Date_ out range month\n");
			SetDateByte(date, val, (MM_POS + base));
			break;
		case DTCD_DD:
			len = 2; 
			LOSStn2Integer(pd, val, len);
			if(len < 0) break;
			if(val < 1 || val > 31) throwExpt(-1, "ExecFormatChar2Date_ out range day\n");
			SetDateByte(date, val, (DD_POS + base));
			break;
		case DTCD_MON:
			if(*++pc != DTCD_DM) { len = -1; break; }
			p = strchr(pd, *++pc);//delimiter문자까지의 주소 획득, pc는 delimiter까지 소화됨
			c = *p;//바로 밑에서 널로 대체되는 문자 저장
			*p = '\0';//delimiter문자자리에 널문자 설정
			rv = name2Month(pd);//달이름에 해당하는 인덱스 획득
			if(rv < 0) { len = -1; break; }
			val = rv;
			if(val < 1 || val > 12) throwExpt(-1, "ExecFormatChar2Date_ out range month\n");
			pd = p +1;//소스포인터를 delimiter문자 다음 옵셋으로 이동
			SetDateByte(date, val, (MM_POS + base));
			*p = c;//위에서 저장된 문자 복원
			break;
		case DTCD_DM:
			if(*pd++ != *++pc) len = -1;
			break;
		case DTCD_H12:
			len = 2;
			LOSStn2Integer(pd, val, len);
			if(len < 0) break;
			save_val = val;
			if(val < 0 || val > 12) {
				throwExpt(-1, "ExecFormatChar2Date_ out range 12hour\n");
			}
			SetDateByte(date, val, (HH_POS + base));
			break;
		case DTCD_H24:
			len = 2; 
			LOSStn2Integer(pd, val, len);
			if(len < 0) break;
			if(val < 0 || val > 24) throwExpt(-1, "ExecFormatChar2Date_ out range 24hour\n");
			SetDateByte(date, val, (HH_POS + base));
			break;
		case DTCD_MI:
			len = 2; 
			LOSStn2Integer(pd, val, len);
			if(len < 0) break;
			if(val < 0 || val > 60) throwExpt(-1, "ExecFormatChar2Date_ out range minute\n");
			SetDateByte(date, val, (MI_POS + base));
			break;
		case DTCD_SS:
			len = 2; 
			LOSStn2Integer(pd, val, len);
			if(len < 0) break;
			if(val < 0 || val > 60) throwExpt(-1, "ExecFormatChar2Date_ out range second\n");
			SetDateByte(date, val, (SS_POS + base));
			break;
		case DTCD_S_YR:
			pc++;//년도를 획득하기위해
			memcpy(&year, pc, 2);
			SetDateByte(date, (unit)year, (YY_POS + base));
			//val = *(ushortx *)pc;
			//SetDateByte(date, val, (YY_POS + base));
			pc += 2;
			continue;
		case DTCD_S_MM:
			pc++;//월을 획득하기위해
			SetDateByte(date, (unit)*pc, (MM_POS + base));
			pc++;
			continue;
		case DTCD_S_DD:
			pc++;//일을 획득하기위해
			SetDateByte(date, (unit)*pc, (DD_POS + base));
			pc++;
			continue;
		case DTCD_S_H24:
			val = 0; pos = HH_POS; goto EFC2D_LA2;
		case DTCD_S_MI:
			val = 0; pos = MI_POS; goto EFC2D_LA2;
		case DTCD_S_SS:
			val = 0; pos = SS_POS; goto EFC2D_LA2;
EFC2D_LA2:;
			SetDateByte(date, val, (pos + base));
			break;
		case DTCD_AM:
			if(save_val > 12) throwExpt(-1, "ExecFormatChar2Date_ hh12 format over value\n");
			if(!strncmp(pd, ((NLSHourFactor *)hour_fac)->amFmtChar, ((NLSHourFactor *)hour_fac)->amFmtLen)) pd += ((NLSHourFactor *)hour_fac)->amFmtLen;
			else if(!strncmp(pd, ((NLSHourFactor *)hour_fac)->pmFmtChar, ((NLSHourFactor *)hour_fac)->pmFmtLen)) {
				pd += ((NLSHourFactor *)hour_fac)->pmFmtLen;
				save_val += 12;//오후이면 12을 더한다.
			} else throwExpt(-1, "ExecFormatChar2Date_ am format value invalid\n");
			ResetDateByte(date, val, (HH_POS + base));
			SetDateByte(date, save_val, (HH_POS + base));
			break;
		default:
			len = -1;
			break;
		}
		if(len < 0) throwExpt(-1, "ExecFormatChar2Date invalid date input\n");
		pc++;
	}
}
#define ExecFormatChar2Date(hour_fac, fmtcode, strdt, date) {\
	if(IS_DATE_FMT(*fmtcode)) ExecFormatChar2Date_(hour_fac, fmtcode, strdt, *(uintx *)date, -4);\
	else if(IS_DTIME_FMT(*fmtcode)) ExecFormatChar2Date_(hour_fac, fmtcode, strdt, *(unit *)date, 0);\
	else ExecFormatChar2Date_(hour_fac, fmtcode, strdt, *(uintx *)date, 0);\
}
template <class a_t> void changeBOrdYear(a_t sor, intx idx)
{
	a_t val = GetDateByte(sor, idx, 16);
	a_t tar;

	CvtByteOrd((bytex *)&val, (bytex *)&tar, sizeof(a_t));
	SetDateByte(sor, tar, idx);
}
/******************************************* to char **********************************/
//prec - 널이 포함된 타겟 전체 길이, rsize - 널이 포함된 head 숫자 스트링 길이, head - 숫자스트링 시작 포인터
//rear - 반올림을 해야하면 prec사이즈 맨끝수의 포인터가 설정되고 할 필요없으면 널이 안에서 설정됨
//타겟 길이가 스트링 길이보다 작으면 숫자 스트링 시작부터 제한된 길이까지 '.'가 없으면 제한된 길이까지의 
//숫자가 소수점 상위 숫자이므로 오버플로우이다. prec가 2일때 123. 이면 오버플로우, 12.3이면 끝에 체크에서 
//*p가 '.'이므로 오버 플로우가 아님
#define CheckOverflow(prec, rsize, head, rear) {\
	bytex *p = head;\
	bytex *end;\
	if(rsize <= prec) rear = nullx;\
	else {\
		for(end = head + (prec -1);p < end; p++) {\
			if(*p == '.') break;\
		}\
		if(p == end) throwExpt(-1, "CheckOverflow overflow\n");\
		else rear = --end;\
	}\
}
//incexp - 1로 설정되면 exp하나 증가됨을 나타낸다. head - 시작 포인터(123.456이면 1을 
//가리키는 포인터), incexp가 1이면 head포인터는 하나더 왼쪽으로 이동된다.
//rear - 소수점 이하 반올림할(scale로 재한된) 넘버스트링의 맨끝수, 이 뒤에 또 수가 있으면 반올림 처리한다.
//overit - 소수점 이상의 수에서는 현재 수가 0가 되고 올림된다해도 rear를 앞으로 옮기지
//			않게 하고 소수이하의 수에서만 rear를 앞으로 옮긴다.
#define HalfOverIt(head, rear, incexp, sp, overit) {\
	bytex *hp = (*head == '+' || *head == '-' ? head +1 : head);\
	overit = 0;\
	for(sp = rear; hp <= sp; sp--) {\
		if(*sp == '9') {\
			*sp = '0';\
			if(overit == 0) rear = sp;\
		} else if(*sp != '.') { *sp += 1; break; }\
		else overit = 1;\
	}\
	if(*rear == '0') *(rear +1) = '\0';\
	if(hp > sp) {\
		if(*sp == '+' || *sp == '-') { head = sp -1; *head = *sp; }\
		else head = sp;\
		*sp = '1';\
		incexp = 1;\
	}\
}
#define HalfOver(head, rear, incexp) {\
	bytex *sp = rear;\
	intx overit = 0;\
	for(++sp; *sp != '\0'; sp++) {\
		if(*sp >= '5') {\
			overit = 1;\
			break;\
		} else if(*sp < '4') break;\
	}\
	if(overit) HalfOverIt(head, rear, incexp, sp, overit);\
}
#define HalfOverEnding(head, rear, incexp) {\
	HalfOver(head, rear, incexp);\
	*(rear +1) = '\0';\
}
//나중에 데이터 입력과정에서 이 매크로로 자리수 절삭을 수행한후 넘버 혹은 이진 타입을 변환한다.
//prec - 전체 넘버 자리수, scale - 소수점 이하 자리수, 
//rv - 소수점 이상 자리수(prec - scale)가 실제 넘버의 소수점 이상 길이보다 작으면 -1(에러
//설정), 소수점 이상이 맨 좌측까지 반올림되어 자리수가 하나 늘어나면 1설정, 그도저도 아니면 0설정.
//rv2 - 데이터의 소수점 이하 길이가 scale보다 작으면 그 차이 길이를 리턴
//s - 시작 포인터 rv가 1로 설정되면 s포인터는 하나더 왼쪽으로 이동된다. 
//p - 넘버가 scale에 의해 잘리든 안잘리든 넘버의 맨끝 널 문자를 가리키는 포인터
#define AdjustPosition(s, p, prec, scale, rv, rv2) {\
	intx nover = prec - scale;\
	intx i;\
	if(*s == '+' || *s == '-') p = s +1;\
	else p = s;\
	for(i = 0;i < nover; i++, p++) {\
		if(*p == '.' || *p == '\0') break;\
	}\
	if(i == nover && (*p != '.' && *p != '\0')) rv = -1;/*k473so.*p != '\0') rv = -1;*/\
	else {\
		rv = i = 0;\
		if(*p == '.') {\
			for(++p; i < scale; i++, p++) {\
				if(*p == '\0') break;\
			}\
			if(i == scale) {\
				p--;\
				HalfOver(s, p, rv);\
				if(*p == '.') *p = '\0';\
				else *++p = '\0';\
			}\
		}\
		if(scale) rv2 = scale - i;\
	}\
}
/*
//만약 s의 소수점 이하가 있은데 scale가 0이면 소수점이하를 잘라버린다.
#define AdjustPosition(s, p, prec, scale, rv) {\
	intx nover = prec - scale;\
	intx i;\
	if(*s == '+' || *s == '-') s++;\
	p = s;\
	for(i = 0;i < nover; i++, p++) {\
		if(*p == '.' || *p == '\0') break;\
	}\
	if(i == nover && *p != '\0') rv = -1;\
	else {\
		rv = 0;\
		if(*p == '.') {\
			if(scale) {\
				for(++p, i = 0; i < scale; i++, p++) {\
					if(*p == '\0') break;\
				}\
				if(i == scale) {\
					p--;\
					HalfOver(s, p, rv);\
					*++p = '\0';\
				}\
			} else *p = '\0';\
		}\
	}\
}
*/
//rear - 넘버스트링의 맨끝 널문자를 지칭하는 포인터, 하나 앞으로 땡겨저서 이동된다.
//예) 10.0이면 1.0되고 rear 포인터는 1.0의 맨끝 널문자를 포인트한다.
#define RearToLeft(rear) {\
	*(rear -3) = *(rear -2);\
	*(rear -2) = *(rear -1);\
	*(rear -1) = '\0';\
	rear--;\
}
//p로부터 exp를 E+/-xxx의 포맷으로 설정한다. 즉 p의 자리에 'E'가 자리한다.
#define PutIEExp(p, exp) {\
	*p++ = 'E';\
	if(exp < 0) { *p++ = '-'; exp *= -1; }\
	else *p++ = '+';\
	*p++ = '0' + exp / 100;\
	exp %= 100;\
	*p++ = '0' + exp / 10;\
	*p++ = '0' + exp % 10;\
	*p = '\0';\
}
//scale +1에 널문자를 삽입한다. 소수점 이하 자리수가 scale보다 작으면 아무것도 안함
#define TruncPosition(s, p, scale) {\
	intx i;\
	for(p = s;*p != '\0'; p++) {\
		if(*p == '.') break;\
	}\
	if(*p == '.') {\
		if(scale) {\
			for(i = 0, p++;*p != '\0' && i < scale; p++, i++);\
		}\
		*p = '\0';\
	}\
}
//소수점뒤에 0가 아닌 숫자가 있으면 소수점 이상 수를 하나 증가된 값을 설정한다.
#define CeilReal(s) {\
	bytex *sp, *p, *rear = s;\
	intx overit;\
	for(;*rear != '\0'; rear++) {\
		if(*rear == '.') break;\
	}\
	if(*rear == '.') {\
		for(p = rear +1;*p != '\0'; p++) {\
			if(*p != '0') break;\
		}\
		if(*p != '\0') {\
			HalfOverIt(s, rear, rv, sp, overit);\
			*rear = '\0';\
		}\
	}\
}
//소수점 '.'가 있으면 그 자리에 널문자를 설정한다.
#define FloorReal(s) {\
	bytex *p = s;\
	for(;*p != '\0'; p++) {\
		if(*p == '.') break;\
	}\
	*p = '\0';\
}

#define CHCD_YR		0
#define CHCD_MM		1
#define CHCD_DD		2
#define CHCD_MONTH	3
#define CHCD_DAY	4
#define CHCD_DTH	5
#define CHCD_H12	6
#define CHCD_H24	7
#define CHCD_MI		8
#define CHCD_SS		9
#define CHCD_PM		10
#define CHCD_AM		11
#define CHCD_DM		12
#define CHCD_NUM	13
#define CHCD_EEE	14
#define CHCD_FM		15
#define CHCD_END	16

typedef struct {
	sytex chNumCd;
	sytex bkLead;
	sytex cntComma;
	sytex chDol;
	sytex chNumLen;
	sytex chUndLen;
	bytex pad[2];
} ChCdFmt;
//포맷이 0999~일경우 실넘버길이가 포맷보다 적을경우 앞을 0로 채우므로 포맷 전체길이를 리드 길이로
//숫자포맷이 00999,99와 같은 경우 전체 포맷 길이 8이고 999,99이면 컴마만의 길이 1
#define GetLeadLength(lead, num_fmt_len, cnt_comma, dol) (lead ? num_fmt_len + cnt_comma + dol: cnt_comma + dol)
//sor - date타입, number타입, 넘버스트링, tar -  내부에서 선두포인터가 변경될수있으므로 참조형으로
//base - sor가 데이트타입이면 ExecFormatChar2Date와 동일, 
//			넘버타입이면 COLUMN_T_INT ~ COLUMN_T_NUMBER, 넘버스트링(예,"1234")이면 COLUMN_T_CHAR
//rsize - sor가 숫자/날짜 타입이면 변환된 넘버스트링 (널 포함된)길이, 그외는 의미없고 값의 변화도 없음
//		(데이트 타입은 변환된 스트링 길이도 소스의 길이와 같고 문자에서 문자는 변환되지않으므로 
//		길이의 변화가 없음)
//cut - 변환대상이 숫자일 경우 절삭을 하라는 의미, 나머지 타입은 의미없음, 이값이 0이면 넘버가 
//		아닌 네이티브 이진수의 디폴트 변환인 경우로 문자열로 변환할때 자리수 절삭이 필요없으므로 
//		절삭을 하지 말라는 의미(네이티브 이진수를 포맷을 주어 변환하는 경우는 포맷에 따라 자리수
//		가 결정되므로 절삭을 할때가 있다.)
template <class a_t> void ExecFormat2Char(bytex *fmtcode, a_t sot_t, bytex *sor, bytex *&tar, intx base, 
										  intx &rsize, intx cut) 
{
	bytex *pc = fmtcode;
	bytex *pd = tar;
	bytex slot[100];
	bytex *p, *pc_head, *pc_end, *tar_head, *tar_end;
	a_t val;
	intx rv, rv2, len, i, numlen, undlen, exp, ibyte, fm = 0, lead_len;
	longx lexp;
	bytex zero[] = {'0', '0', '0', '0', '\0'};
	intx year, month, day;
	ChCdFmt *chcd_fmt;

	for(;*pc != CHCD_END;) {
		val = 0;
		switch(*pc) {
		case CHCD_YR:
			ibyte = YY_POS + base;
			if(ibyte < 0 || ibyte >= sizeof(a_t)) p = zero;
			else {
				val = GetDateByte(*(a_t *)sor, ibyte, 16);
				Long2Stn(val, slot, len, p);
			}
			p += *++pc;//(4 - YY count)
			for(;*p != '\0'; pd++, p++) *pd = *p;
			break;
		case CHCD_MM:
			ibyte = MM_POS + base; 
			if(ibyte < 0 || ibyte >= sizeof(a_t)) val = 1;
			else val = GetDateByte(*(a_t *)sor, ibyte, 8);
			goto EFC2_LA1;
		case CHCD_DD:
			ibyte = DD_POS + base; 
			if(ibyte < 0 || ibyte >= sizeof(a_t)) val = 1;
			else val = GetDateByte(*(a_t *)sor, ibyte, 8);
			goto EFC2_LA1;
		case CHCD_MONTH:
			ibyte = MM_POS + base;
			if(ibyte < 0 || ibyte >= sizeof(a_t)) val = 1;
			else val = GetDateByte(*(a_t *)sor, ibyte, 8);
			for(p = monthName[val -1];*p != '\0'; pd++, p++) *pd = *p;
			break;
		case CHCD_DAY:
			ibyte = DD_POS + base;
			if(ibyte < 0 || ibyte >= sizeof(a_t)) val = 1;
			else val = GetDateByte(*(a_t *)sor, ibyte, 8);
			year = (intx)VGetDateByte(*(a_t *)sor, (YY_POS + base), 16);
			month = (intx)VGetDateByte(*(a_t *)sor, (MM_POS + base), 8);
			day = (intx)VGetDateByte(*(a_t *)sor, (DD_POS + base), 8);
			GetCurWDay(year, month, day, val);
			for(p = dayName[val];*p != '\0'; pd++, p++) *pd = *p;
			break;
		case CHCD_DTH:
			ibyte = DD_POS + base;
			if(ibyte < 0 || ibyte >= sizeof(a_t)) val = 1;
			else val = GetDateByte(*(a_t *)sor, ibyte, 8);
			year = (intx)VGetDateByte(*(a_t *)sor, (YY_POS + base), 16);
			month = (intx)VGetDateByte(*(a_t *)sor, (MM_POS + base), 8);
			day = (intx)VGetDateByte(*(a_t *)sor, (DD_POS + base), 8);
			GetCurWDay(year, month, day, val);
			*pd++ = '1' + (intx)val;
			break;
		case CHCD_H12:
			ibyte = HH_POS + base;
			if(ibyte < 0 || ibyte >= sizeof(a_t)) val = 0;
			else val = GetDateByte(*(a_t *)sor, ibyte, 8);
			if(val > 12) val -= 12;
			else if(val == 0) val = 12;
			*pd++ = (bytex)('0' + val / 10);
			*pd++ = (bytex)('0' + val % 10);
			break;
		case CHCD_H24:
			ibyte = HH_POS + base;
			if(ibyte < 0 || ibyte >= sizeof(a_t)) val = 0;
			else val = GetDateByte(*(a_t *)sor, ibyte, 8);
			*pd++ = (bytex)('0' + val / 10);
			*pd++ = (bytex)('0' + val % 10);
			break;
		case CHCD_MI:
			ibyte = MI_POS + base; 
			if(ibyte < 0 || ibyte >= sizeof(a_t)) val = 0;
			else val = GetDateByte(*(a_t *)sor, ibyte, 8);
			goto EFC2_LA1;
		case CHCD_SS:
			ibyte = SS_POS + base;
			if(ibyte < 0 || ibyte >= sizeof(a_t)) val = 0;
			else val = GetDateByte(*(a_t *)sor, ibyte, 8);
EFC2_LA1:;
			*pd = (bytex)('0' + val / 10);
			if(fm == 0 || *pd != '0') pd++;
			*pd++ = (bytex)('0' + val % 10);
			break;
		case CHCD_PM:
		case CHCD_AM:
			val = GetDateByte(*(a_t *)sor, (HH_POS + base), 8);
			if(val < 12) { len = strleng("오전"); strncpy(pd, "오전", len); }
			else { len = strleng("오후"); strncpy(pd, "오후", len); }
			pd += len;
			break;
		case CHCD_DM:
			*pd++ = *++pc;
			break;
		case CHCD_NUM:
			chcd_fmt = (ChCdFmt *)pc;
			numlen = chcd_fmt->chNumLen;
			undlen = chcd_fmt->chUndLen;
			lead_len = GetLeadLength(chcd_fmt->bkLead, numlen, chcd_fmt->cntComma, chcd_fmt->chDol);
			pc += sizeof(ChCdFmt);
			switch(base) {
				case COLUMN_T_SHORT:
					break;
				case COLUMN_T_FLOAT:
					Float2Stn(*(floatx *)sor, (tar + lead_len), rsize, tar);
					break;
				case COLUMN_T_INT:
					Int2Stn(*(intx *)sor, (tar + lead_len), rsize, tar);//이 매크로는 sor가 변경되므로 주의
					break;
				case COLUMN_T_LONG:
					Long2Stn(*(longx *)sor, (tar + lead_len), rsize, tar);//이 매크로는 sor가 변경되므로 주의
					break;
				case COLUMN_T_DOUBLE:
					Double2Stn(*(doublex *)sor, (tar + lead_len), rsize, tar);
					break;
				case COLUMN_T_NUMBER:
					Number2Char(sor, (tar + lead_len), rsize, tar);
					break;
				case COLUMN_T_CHAR:	//sor이 숫자 스트링이므로 따로 변환할것이 없다.
				case COLUMN_T_VCHAR:
					tar = sor;
					break;
			}
			if(chcd_fmt->cntComma == 0 && cut) {
				AdjustPosition(tar, p, numlen, undlen, rv, rv2);
				if(rv < 0) throwExpt(-1, "ExecFormat2Char number out of length");
				rsize = (intx)(p - tar);//널이 제외된 전체 길이
				if(fm == 0 && undlen && rv2) {//scale이 있고 소수점이하 길이가 scale보다 작으면
					if(undlen == rv2) *p++ = '.';//데이터의 소수점이하가 없으면
					for(i = 0;i < rv2; i++) *p++ = '0';//스케일길이만큼 소수점이하 0로 채움
					*p = '\0';
					rsize += rv2;//모자란 소수점이하를 채운 길이만큼 전체길이 더함
				}
			}
			if(chcd_fmt->cntComma || chcd_fmt->bkLead) {
				pc_head = pc;
				pc_end = pc = pc + (numlen -1) + chcd_fmt->cntComma;//숫자 포맷의 끝
				tar_head = tar;
				tar_end = tar + rsize;//숫자 소스의 끝
				tar = tar_end + lead_len;//새로 타겟의 끝 설정
				*tar = '\0';//타겟의 끝 널마감
				for(;;pc_end--) {//끝에서부터 처음으로
					if(pc_end < pc_head) break;
					if(*pc_end == '9') {
						if(tar_head < tar_end) *--tar = *--tar_end;
						else {
							if(chcd_fmt->bkLead && fm == 0) {
								*--tar = '0';
								rsize++;
							} else break;
						}
					} else {//','
						if(tar_head < tar_end || (chcd_fmt->bkLead && fm == 0)) {
							*--tar = ',';
							rsize++;
						} else break;
					}
				}
			}
			if(chcd_fmt->chDol) {
				*--tar = '$';
				rsize++;//'$'한 문자길이 추가
			}
			rsize++;//널문자 한바이트 추가
			break;
		case CHCD_EEE:
			chcd_fmt = (ChCdFmt *)pc;
			numlen = chcd_fmt->chNumLen;
			undlen = chcd_fmt->chUndLen;
			pc += sizeof(ChCdFmt);
			switch(base) {
				case COLUMN_T_SHORT:
					break;
				case COLUMN_T_FLOAT:
					Float2IEStn(*(floatx *)sor, tar, tar, exp);
					break;
				case COLUMN_T_INT:
					Int2IEStn(*(intx *)sor, tar, tar, exp);
					break;
				case COLUMN_T_LONG:
					Long2IEStn(*(longx *)sor, tar, tar, exp);
					break;
				case COLUMN_T_DOUBLE:
					Double2IEStn(*(doublex *)sor, tar, tar, lexp);
					exp = (intx)lexp;
					break;
				case COLUMN_T_NUMBER:
					Number2IEChar(sor, tar, rsize, tar, exp);
					break;
				case COLUMN_T_CHAR:	//sor이 숫자 스트링이므로 중요하지않으므로 나중에 구현
				case COLUMN_T_VCHAR:
					break;
			}
			if(cut) {
				AdjustPosition(tar, p, numlen, undlen, rv, rv2);
				if(rv < 0) throwExpt(-1, "ExecFormat2Char number out of length");
				if(rv == 1) {
					RearToLeft(p);
					exp++;
				}
			}
			PutIEExp(p, exp);
			rsize = (intx)(p - tar) +1;//+1은 널문자
			return;//break해도 됨
		case CHCD_FM:
			fm = 1;
			break;
		}
		pc++;
	}
	if(pd > tar) {//날짜스트링 변환에서만 pd가 증가하므로 여기는 날짜스트링 변환만 수행됨
		rsize = (intx)(pd - tar) +1;//+1은 널문자
		*pd = '\0';//날짜스트링 변환은 널로 마감안됐으므로 여기서 마감
	}
}

#define GET_MANTISA_MASK(bmsk, bits, lenMan, type) bmsk = ~((type)0) >> (bits - lenMan)
#define GET_SIGN(pd, bits, type) *(type *)pd >> (bits -1)
#define GET_USIGN(pd, bits, type) (intx)(GET_SIGN(pd, bits, type))

#define MASKING_MANTISA(man, pd, bmsk) man = *pd & bmsk
#define GET_EXP(pd, bmsk, lenMan, lenExp) ((*pd & ~bmsk) >> lenMan) & ~(1 << lenExp)
#define GET_UEXP(pd, bmsk, lenMan, lenExp) (intx)(GET_EXP(pd, bmsk, lenMan, lenExp))

#define ALIVE_HIDDEN(ah, mantisa, lenMan, exp, type) {\
	if(ah) {/*ud의 가수부가 히든비트되어있고 히든비트를 살려낸다.*/\
		mantisa |= ((type)1 << lenMan);/*가수부에 hidden bit를 추가한다.*/\
		exp++;/*123.456이 ieee는 1.2345(biase + 2), hidden되지않은경우는 0.12345(biase + 3)|exp < biase경우도 마찬가지*/\
	}\
}
#define SET_MANTISA(pd, bmsk, man) *pd = bmsk & man
#define SET_EXP(pd, exp, lenMan) *pd = *pd | (exp << lenMan)
#define SET_SIGN(pd, sign, lenMan, lenExp) *pd = *pd | (sign << (lenMan + lenExp))
#define HIDDEN_PROCESS(hid, exp, bmsk) {\
	if(hid) {/*주어진 지수나 가수는 히든비트하지 않은 상태*/\
		exp--;/*주어진 지수는 0.1xxx, 히든 처리하면 1.xxx이므로 하나감소*/\
		bmsk >>= 1;/*가수의 맨앞 비트 1을 제거하기위해*/\
	}\
}
#define TAKEOUT_SEM32(pd, lenExp, lenMan, alive_hidden, sign, exp, mantisa) {\
	uintx b = 0;\
	GET_MANTISA_MASK(b, 32, lenMan, uintx);\
	MASKING_MANTISA(mantisa, pd, b);\
	exp = GET_EXP(pd, b, lenMan, lenExp);\
	sign = GET_SIGN(pd, 32, uintx);\
	ALIVE_HIDDEN(alive_hidden, mantisa, lenMan, exp, intx);\
}
#define TAKEIN_SEM32(pd, lenExp, lenMan, hidden, sign, exp, mantisa) {\
	uintx b;\
	GET_MANTISA_MASK(b, 32, lenMan, uintx);\
	HIDDEN_PROCESS(hidden, exp, b);\
	SET_MANTISA(pd, b, mantisa);\
	SET_EXP(pd, exp, lenMan);\
	SET_SIGN(pd, sign, lenMan, lenExp);\
}
#define TAKEOUT_SEM64(pd, lenExp, lenMan, alive_hidden, sign, exp, mantisa) {\
	unit b = 0;\
	GET_MANTISA_MASK(b, 64, lenMan, unit);\
	MASKING_MANTISA(mantisa, pd, b);\
	exp = GET_UEXP(pd, b, lenMan, lenExp);\
	sign = GET_USIGN(pd, 64, unit);\
	ALIVE_HIDDEN(alive_hidden, mantisa, lenMan, exp, unit);\
}
#define TAKEIN_SEM64(pd, lenExp, lenMan, hidden, sign, exp, mantisa) {\
	unit b;\
	GET_MANTISA_MASK(b, 64, lenMan, unit);\
	HIDDEN_PROCESS(hidden, exp, b);\
	SET_MANTISA(pd, b, mantisa);\
	SET_EXP(pd, (unit)exp, lenMan);\
	SET_SIGN(pd, (unit)sign, lenMan, lenExp);\
}

#define BIASE_REV(exp, sor_bias, tar_bias) ((exp - sor_bias) + tar_bias)


#endif
