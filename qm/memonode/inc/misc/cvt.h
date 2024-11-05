#ifndef _H_CVT
#define _H_CVT

#include "misc/xt.h"
#include "misc/basys.h"
/* pointer  number(int, uint, float) to string은 string ptr을 증가하지않는다. */
/* pointer string to number은 string ptr을 증하한다 */
/* array number to str, array str to number은 당연히 증가하지않는다. */
#define INTCNSTN(n, s) {\
	int i41, j41, len41, c41;\
	unit n41, sign41;\
	\
	n41 = n;\
	if((sign41 = n41) < 0) n41 = -n41;\
	i41 = 0;\
	do {\
		s[i41++] = (bytex)(n41 % 10 + '0');\
	} while ((n41 /= 10) > 0);\
	if(sign41 < 0) s[i41++] = '-';\
	s[i41] = '\0';\
	\
/*	i41 = 0; */\
/*	while(s[i41] != '\0') ++i41; */\
	len41 = i41;\
	\
	for(i41 = 0, j41 = len41-1; i41 < j41; i41++, j41--) {\
		c41 = s[i41];\
		s[i41] = s[j41];\
		s[j41] = c41;\
	}\
	\
}


#define UINTCNSTN(n, s) {\
	unsigned int i40, j40, len40, c40;\
	unit n40;\
	\
	n40 = n;\
	i40 = 0;\
	do {\
		s[i40++] = (bytex)(n40 % 10 + '0');\
	} while ((n40 /= 10) > 0);\
	s[i40] = '\0';\
	\
/*	i40 = 0; */\
/*	while(s[i40] != '\0') ++i40; */\
	len40 = i40;\
	\
	for(i40 = 0, j40 = len40-1; i40 < j40; i40++, j40--) {\
		c40 = s[i40];\
		s[i40] = s[j40];\
		s[j40] = c40;\
	}\
	\
}

#define STNCNINT(s, rn) {\
	int i39, n39, sign39;\
	unit n39;\
	for(i39 = 0; (s[i39] == ' ') || (s[i39] == '\n') || (s[i39] == '\t') || (s[i39] == '\r'); i39++) ;\
	sign39 = (s[i39] == '-') ? -1 : 1;\
	if(s[i39] == '+' || s[i39] == '-') i39++;\
	for(n39 = 0; (s[i39] >= '0') && (s[i39] <= '9'); i39++)\
		n39 = 10 * n39 + (s[i39] - '0');\
	rn = sign39 * n39;\
}

#define STNCNUINT(s, n) {\
	int i38;\
	\
	n = 0;\
	for(i38 = 0; (s[i38] >= '0') && (s[i38] <= '9'); ++i38)\
		n = 10 * n + (s[i38] - '0');\
}



#define PINTCNSTN(n, s) {\
	int i37, j37, len37, c37;\
	unit n37, sign37;\
	n37 = n;\
	if((sign37 = n37) < 0) n37 = -n37;\
	i37 = 0;\
	do {\
		*(s + i37++) = (bytex)(n37 % 10 + '0');\
	} while ((n37 /= 10) > 0);\
	if(sign37 < 0) *(s + i37++) = '-';\
	*(s + i37) = '\0';\
	\
/*	i37 = 0; */\
/*	while(*(s + i37) != '\0') ++i37; */\
	len37 = i37;\
	\
	for(i37 = 0, j37 = len37-1; i37 < j37; i37++, j37--) {\
		c37 = *(s + i37);\
		*(s + i37) = *(s + j37);\
		*(s + j37) = c37;\
	}\
	\
}


#define PUINTCNSTN(n, s) {\
	unsigned int i36, j36, len36, c36;\
	unit n36;\
	n36 = n;\
	i36 = 0;\
	do {\
		*(s + i36++) = (bytex)(n36 % 10 + '0');\
	} while ((n36 /= 10) > 0);\
	*(s + i36) = '\0';\
	\
/*	i36 = 0; */\
/*	while(*(s + i36) != '\0') ++i36; */\
	len36 = i36;\
	\
	for(i36 = 0, j36 = len36-1; i36 < j36; i36++, j36--) {\
		c36 = *(s + i36);\
		*(s + i36) = *(s + j36);\
		*(s + j36) = c36;\
	}\
	\
}


#define PSTNCNINT(s, rn) {\
	int sign35;\
	unit n35;\
	if(s) {\
		for(; (*s == ' ') || (*s == '\n') || (*s == '\t') || (*s == '\r'); s++) ;\
		sign35 = (*s == '-') ? -1 : 1;\
		if(*s == '+' || *s == '-') s++;\
		for(n35 = 0; (*s >= '0') && (*s <= '9'); s++)\
			n35 = 10 * n35 + (*s - '0');\
		rn = sign35 * n35;\
	} else rn = 0;\
}
#define PSTNCNUINT(s, n) {\
	n = 0;\
	if(s) {\
		for(; (*s >= '0') && (*s <= '9'); ++s)\
			n = 10 * n + (*s - '0');\
	} else n = 0;\
}

#define PFLOATCNSTN(d, s) sprintf(s, "%f", d);

#define PSTNCNFLOAT(s, d) {\
	double val33, power33;\
	int sign33;\
	if(s) {\
		for(; (*s == ' ') || (*s == '\n') || (*s == '\t') || (*s == '\r'); s++) ;\
		sign33 = (*s == '-') ? -1 : 1;\
		if(*s == '+' || *s == '-') s++;\
		for(val33 = 0.0; *s >= '0' && *s <= '9'; s++) val33 = 10.0 * val33 + (*s - '0');\
		if(*s == '.') s++;\
		for(power33 = 1.0; (*s >= '0') && (*s <= '9'); s++) {\
			val33 = 10.0 * val33 + (*s - '0');\
			power33 *= 10.0;\
		}\
		d = sign33 * val33 /power33;\
	} else d = 0;\
}


//length outskirt
#define LOSStn2Integer(s, d, len) {\
	intx sign, i;\
	if(s) {\
		for(; (*s == ' ') || (*s == '\n') || (*s == '\t') || (*s == '\r'); s++) ;\
		sign = (*s == '-' ? -1 : 1);\
		if(*s == '+' || *s == '-') s++;\
		for(d = 0, i = 0; i < len; s++, i++) {\
			if((*s >= '0') && (*s <= '9')) d = 10 * d + (*s - '0');\
			else { len = -1; break; }\
		}\
		if(sign == -1) d = sign * d;\
	} else d = 0;\
}
#define Stn2Integer(s, d) {\
	intx sign;\
	if(s) {\
		for(; (*s == ' ') || (*s == '\n') || (*s == '\t') || (*s == '\r'); s++) ;\
		sign = (*s == '-' ? -1 : 1);\
		if(*s == '+' || *s == '-') s++;\
		for(d = 0; (*s >= '0') && (*s <= '9'); s++) d = 10 * d + (*s - '0');\
		if(sign == -1) d = sign * d;\
	} else d = 0;\
}
#define Hex2Integer(s, d) {\
	intx sign;\
	if(s) {\
		for(; (*s == ' ') || (*s == '\n') || (*s == '\t') || (*s == '\r'); s++) ;\
		sign = (*s == '-' ? -1 : 1);\
		if(*s == '+' || *s == '-') s++;\
		for(d = 0;; s++) {\
			if((*s >= '0') && (*s <= '9')) d = 16 * d + (*s - '0');\
			else if((*s >= 'a') && (*s <= 'f'))	d = 16 * d + (10 + (*s - 'a'));\
			else if((*s >= 'A') && (*s <= 'F')) d = 16 * d + (10 + (*s - 'A'));\
			else break;\
		}\
		if(sign == -1) d = sign * d;\
	} else d = 0;\
}
#define Stn2Real(tp, s, d) {\
	tp power;\
	intx sign;\
	if(s) {\
		for(; (*s == ' ') || (*s == '\n') || (*s == '\t') || (*s == '\r'); s++);\
		sign = (*s == '-' ? -1 : 1);\
		if(*s == '+' || *s == '-') s++;\
		for(d = 0.0; *s >= '0' && *s <= '9'; s++) d = (tp)(10.0 * d + (*s - '0'));\
		if(*s == '.') s++;\
		for(power = 1; (*s >= '0') && (*s <= '9'); s++) {\
			power *= (tp)0.1;\
			d = d + (*s - '0') * power;\
		}\
		if(sign == -1) d = sign * d;\
	} else d = 0.0;\
}
#define Stn2Float(s, d) Stn2Real(floatx, s, d)
#define Stn2Double(s, d) Stn2Real(doublex, s, d)
//rsize - 끝의 널문자를 제외한 변한된 숫자 스트링 길이
#define Real2Stn(otp, utp, bit, lead, lim, d, si, rsize, so) {\
	intx sign, i;\
	otp bover, exp;\
	utp bunder, dd = d;\
	bytex *_p;\
	DevideMantisa##bit(dd, sign, exp, bover, bunder);\
	so = si + lead;\
	_p = so +1;\
	for(;bover;) {\
		*so-- = (bytex)(bover % 10 + '0');\
		bover /= 10;\
	}\
	if(++so == _p) *--so = '0';\
	if(sign == 1) *--so = '-';\
	if(bunder) *_p++ = '.';\
	for(i = 0;bunder && i < lim; i++) {\
		bunder *= 10;\
		*_p++ = ((intx)bunder + '0');\
		bunder = bunder - (intx)bunder;\
	}\
	*_p = '\0';\
	rsize = (intx)(_p - so);\
}
#define FLT_LEAD	10
#define FLT_LIM		7
#define FLT2STR_LEN	19
#define DBL_LEAD	20
#define DBL_LIM		15
#define DBL2STR_LEN	37
//si버퍼의 크기는 lead + lim이어야 한다.
#define FLOAT_LEAD	10
#define FLOAT_LIM	30
#define FLOAT2STR_LEN	40
#define DOUBLE_LEAD 20
#define DOUBLE_LIM	40	//원래는 324바이트 이상이어야 하나 효율상
#define DOUBLE2STR_LEN	64
//si(50 byte) - 출력 문자버퍼의 시작포인터(47 byte이상 - 부호1byte + 가수의 최대길이 
//7(2의 23승)자리 + 지수의 소수점 하위 최대길이 MAX_SCALE(2의 127승) + 널값 1byte), 
//so - 출력 변환된 숫자문자열의 시작포인터로서 일단 중간으로 이동시키는 lead옵셋(
//8 byte outer - 부호 + 가수최대길이)
#define Float2Stn(d, si, rsize, so) Real2Stn(intx, floatx, 32, FLOAT_LEAD, FLOAT_LIM, d, si, rsize, so)
#define _Float2Stn(lead, lim, d, si, rsize, so) Real2Stn(intx, floatx, 32, lead, lim, d, si, rsize, so)
//si(330 byte) - 출력 문자버퍼의 시작포인터(324 byte이상 - 부호1byte + 가수의 최대길이 
//16(2의 53승)자리 + 지수의 소수점 하위 최대길이 307(2의 1023승) + 널값 1byte), 
//so - 출력 변환된 숫자문자열의 시작포인터로서 일단 중간으로 이동시키는 lead옵셋(
//17 byte outer - 부호 + 가수최대길이)
#define Double2Stn(d, si, rsize, so) Real2Stn(longx, doublex, 64, DOUBLE_LEAD, DOUBLE_LIM, d, si, rsize, so)
#define _Double2Stn(lead, lim, d, si, rsize, so) Real2Stn(longx, doublex, 64, lead, lim, d, si, rsize, so)
#define INT_LEAD	15
#define INT2STR_LEN	16
#define LONG_LEAD	23
#define LONG2STR_LEN 24
//si - 출력 문자버퍼의 시작포인터(22byte이상 - 부호 1바이트 + 최대 유닛사이즈 정수 
//2의 64승 20자리 + 1byte 널값), so - 출력 변환된 숫자문자열의 시작포인터, 
//최대 유닛사이즈 정수 2의 64승이므로 20자리, so - 출력 변환된 숫자문자열의 시작포인터로서 
//일단 끝으로 이동시키는데 그 lead는 사이즈만큼, rsize - 끝의 널문자를 제외한 변한된 숫자 스트링 길이
#define Integer2Stn(otp, lead, d, si, rsize, so) {\
	otp sign = (d < 0 ? 1 : 0), dd = d;\
	bytex *_p;\
	if(dd < 0) dd *= -1;\
	so = si + lead;\
	*--so = '\0';\
	_p = so;\
	for(;dd;) {\
		*--so = (bytex)(dd % 10 + '0');\
		dd /= 10;\
	}\
	if(*so == '\0') *--so = '0';\
	if(sign) *--so = '-';\
	rsize = (intx)(_p - so);\
}
#define Int2Stn(d, si, rsize, so) Integer2Stn(intx, INT_LEAD, d, si, rsize, so)
#define Long2Stn(d, si, rsize, so) Integer2Stn(longx, LONG_LEAD, d, si, rsize, so)

#define Integer2Hex(otp, lead, d, si, rsize, so) {\
	otp sign = (d < 0 ? 1 : 0), dd = d;\
	bytex *_p;\
	intx n;\
	if(dd < 0) dd *= -1;\
	so = si + lead;\
	*--so = '\0';\
	_p = so;\
	for(;dd;) {\
		n = dd % 16;\
		if(n < 10) *--so = (bytex)(n + '0');\
		else {\
			n -= 10;\
			*--so = (bytex)(n + 'a');\
		}\
		dd /= 16;\
	}\
	if(*so == '\0') *--so = '0';\
	if(sign) *--so = '-';\
	rsize = (intx)(_p - so);\
}
#define Int2Hex(d, si, rsize, so) Integer2Hex(intx, INT_LEAD, d, si, rsize, so)
#define Long2Hex(d, si, rsize, so) Integer2Hex(longx, LONG_LEAD, d, si, rsize, so)
/*void Integer2Stn(intx i, bytex *si, bytex *&so)
{
	intx sign = (i < 0 ? 1 : 0);
	intx d = (i < 0 ? i * -1 : i);
	so = si + 21;
	*--so = '\0';
	for(;d;) {
		*--so = (d % 10 + '0');
		d /= 10;
	}
	if(*so == '\0') *--so = '0';
	if(sign) *--so = '-';
}*/
#define Real2IEStn(otp, utp, bit, lead, lim, d, si, so, exp) {\
	intx sign, on, overit, underit, opow, upow, i;\
	otp bover, val;\
	utp bunder, dd = d;\
	bytex *_p;\
	\
	so = si + lead;\
	_p = so +1;\
	DevideMantisa##bit(dd, sign, exp, bover, bunder);\
	overit = (bover ? 1 : 0);\
	underit = (bunder ? 1 : 0);\
	on = (underit ? 1 : 0);\
	for(opow = 0;bover; opow++) {\
		val = bover % 10;\
		if(on || val) {\
			*so-- = (bytex)(val + '0');\
			if(val) on = 1;\
		}\
		bover /= 10;\
	}\
	so++;\
	on = (overit ? 1 : 0);\
	for(i = 0, upow = 0;i < lim && bunder;i++) {\
		bunder *= 10;\
		val = (otp)bunder;\
		if(on || val) {\
			*_p++ = (bytex)(val + '0');\
			if(val) on = 1;\
		}\
		if(on == 0) upow--;\
		bunder = bunder - val;\
	}\
	if(so == _p) { *so = '0'; _p++; }\
	*(so -1) = *so;\
	*so = '.';\
	so--;\
	if(sign == 1) *--so = '-';\
	if(*(_p -1) == '0') _p--;\
	if(*(_p -1) == '.') *_p++ = '0';\
	if(overit) exp = opow -1;\
	else if(underit) exp = upow -1;\
	else exp = 0;\
	*_p = '\0';\
}
//매개변수 설명은 Float2Stn와 동일
#define Float2IEStn(d, si, so, exp) Real2IEStn(intx, floatx, 32, FLOAT_LEAD, FLOAT_LIM, d, si, so, exp)
#define Double2IEStn(d, si, so, exp) Real2IEStn(longx, doublex, 64, DOUBLE_LEAD, DOUBLE_LIM, d, si, so, exp)
//si - 출력 문자버퍼의 시작포인터(26byte이상, 1(부호)1byte + 20(2의 64승) + 5('E+000') + 
//1방이트 널값), so - 출력 변환된 숫자문자열의 시작포인터로서 일단 중간으로 이동시키는데 
//그 lead는 21
#define Integer2IEStn(otp, d, si, so, exp) {\
	otp sign = (d < 0 ? 1 : 0), dd = d;\
	intx on, val;\
	bytex *_p;\
	if(dd < 0) dd *= -1;\
	so = si + 21;\
	_p = so +1;\
	for(on = exp = 0;dd; exp++) {\
		val = (intx)(dd % 10);\
		if(on || val) {\
			*so-- = val + '0';\
			if(val) on = 1;\
		}\
		dd /= 10;\
	}\
	so++;\
	if(so == _p) { *so = '0'; _p++; }\
	*(so -1) = *so;\
	*so = '.';\
	so--;\
	if(sign == 1) *--so = '-';\
	if(*(_p -1) == '0') _p--;\
	if(*(_p -1) == '.') *_p++ = '0';\
	if(exp) exp--;\
	*_p = '\0';\
}
#define Int2IEStn(d, si, so, exp) Integer2IEStn(intx, d, si, so, exp)
#define Long2IEStn(d, si, so, exp) Integer2IEStn(longx, d, si, so, exp)
//_sor의 바이트 열을 역순으로 _tar에 write한다.
#define CvtByteOrd(_sor, _tar, bsize) {\
	bytex *_psor = _sor, *_ptar = _tar + (bsize -1), *_pend;\
	for(_pend = _psor + bsize;_psor < _pend; _psor++, _ptar--) *_ptar = *_psor;\
}
#define CvtByteOrd2(_sor, bsize) {\
	bytex *_psor = (bytex *)_sor, *_ptar = (bytex *)_sor + (bsize -1), *_pend, tmp;\
	for(_pend = _psor + bsize/2;_psor < _pend; _psor++, _ptar--) {\
		tmp = *_ptar;\
		*_ptar = *_psor;\
		*_psor = tmp;\
	}\
}
#define PINTCNSTN_LEN(n, s, len) {\
	int i32, j32, c32;\
	unit n32, sign32;\
	\
	n32 = n;\
	if((sign32 = n32) < 0) n32 = -n32;\
	i32 = 0;\
	do {\
		*(s + i32++) = (bytex)(n32 % 10 + '0');\
	} while ((n32 /= 10) > 0);\
	if(sign32 < 0) *(s + i32++) = '-';\
	*(s + i32) = '\0';\
	\
/*	i32 = 0; */\
/*	while(*(s + i32) != '\0') ++i32; */\
	len = i32;\
	\
	for(i32 = 0, j32 = len-1; i32 < j32; i32++, j32--) {\
		c32 = *(s + i32);\
		*(s + i32) = *(s + j32);\
		*(s + j32) = c32;\
	}\
	\
}

#define PUINTCNSTN_LEN(n, s, len) {\
	unsigned int i31, j31, c31;\
	unit n31;\
	\
	n31 = n;\
	i31 = 0;\
	do {\
		*(s + i31++) = (bytex)(n31 % 10 + '0');\
	} while ((n31 /= 10) > 0);\
	*(s + i31) = '\0';\
	\
/*	i31 = 0; */\
/*	while(*(s + i31) != '\0') ++i31; */\
	len = i31;\
	\
	for(i31 = 0, j31 = len-1; i31 < j31; i31++, j31--) {\
		c31 = *(s + i31);\
		*(s + i31) = *(s + j31);\
		*(s + j31) = c31;\
	}\
	\
}

#define PFLOATCNSTN_LEN(d, s, len) {\
	/* mission critical modifiy, (char *)추가*/\
	sprintf((char *)s, "%f", d);\
	len = strleng((char *)s);\
}


#define COMPARE_BYTEX(ndp, snd, cdp, scd, rv) {\
	if(snd > scd) {\
		rv = strncmp(ndp, cdp, scd);\
		if(rv == 0) rv = 1;\
	} else if(snd < scd) {\
		rv = strncmp(ndp, cdp, snd);\
		if(rv == 0) rv = -1;\
	} else rv = strncmp(ndp, cdp, snd);\
}

#define NCOMPARE_BYTEX(ndp, cdp, n, rv) {rv = strncmp(ndp, cdp, n); }
//cdp가 ndp보다 짧아서 중간에 '\0'이 비교된다 하더라도 틀리므로 루프를 벗어나므로 그 이상 
//포인터가 진행되는 일은 없다
/*#define COMPARE_BYTEX(ndp, snd, cdp, scd, rv) {\
	if(snd > scd) rv = 1;\
	else if(snd < scd) rv = -1;\
	else if(snd > PERFORM_CONST) rv = strncmp(ndp, cdp, snd);\
	else {\
		register intx i_;\
		register bytex *np_, *cp_;\
		np_ = ndp; cp_ = cdp;\
		for(i_ = rv = 0; i_ < snd; i_++, np_++, cp_++) {\
			if(*np_ > *cp_) {rv = 1; break;}\
			else if(*np_ < *cp_) {rv = -1; break;}\
		}\
	}\
}
#define NCOMPARE_BYTEX(ndp, cdp, n, rv) {\
	if(n > PERFORM_CONST) rv = strncmp(ndp, cdp, n);\
	else {\
		register intx i_;\
		register bytex *np_, *cp_;\
		np_ = ndp; cp_ = cdp;\
		for(i_ = rv = 0; i_ < n; i_++, np_++, cp_++) {\
			if(*np_ > *cp_) {rv = 1; break;}\
			else if(*np_ < *cp_) {rv = -1; break;}\
		}\
	}\
}
*/

#define V_COLON  	':'
#define V_STAR		'*'
#define V_COLON_STR	":"
#define V_STAR_STR	"*"

#define P_COLON(p, sz, i, j) {\
	bytex *sp_ = p;\
	for(i = 0; i < sz && *p != V_COLON; p++, i++);\
	if(*p != V_COLON) {i = 0; p = sp_; j = sz;}\
	else {p++; j = sz - (i + 1);}/*colon이후의 남은 크기, +1은 :*/\
}
#define P_COLON2(p, sz, i, j) {\
	bytex *sp_ = p;\
	for(i = 0; i < sz && *p != V_COLON; p++, i++);\
	if(*p != V_COLON) {i = 0; p = sp_; }\
	j = sz - i;/*colon포함 이후의 남은 크기*/\
}

//minus이면 2번째 비트에 1 마킹
#define MARK_NUM_PLUS(b) *b = ~((bytex)1 << 1) & (bytex)*b
#define MARK_NUM_MINUS(b) *b = ((bytex)1 << 1) | (bytex)*b
#define NUM_MINUS(b) (((bytex)1 << 1) & (bytex)b)
//정수형 number이면 1번째 비트에 1 마킹
#define MARK_NUM_FLOATING(b) *b = ~(bytex)1 & (bytex)*b
#define MARK_NUM_INTEGER(b) *b = (bytex)1 | (bytex)*b
#define NUM_INTEGER(b) ((bytex)1 & (bytex)b)
// +2은 exp(2)
#define OFFSET_NUM_SIGN(p) ((bytex *)p + 2)
#define REVERSE_NUM_SIGN(p) (NUM_MINUS(*p) ? MARK_NUM_PLUS(p) : MARK_NUM_MINUS(p))

#define MARK_NUM_EXP(p, exp) *(shortx *)p = (shortx)(exp)
#define NUM_EXP(p) (*(shortx *)p)

#ifdef __cplusplus
extern "C" {
#endif
//#include "grm.h"
struct Session_;
//같은가 틀린가만 비교한다.
inline intx isequal(bytex *ndp, intx snd, bytex *cdp, intx scd)
{
	return (snd == scd && !strncmp(ndp, cdp, snd)) ? 1 : 0;
}
extern intx unitcvtstn(unit u, bytex *dp);
extern void stncvtunit(bytex *sp, unit *u); 
extern doublex numcvtfloat(intx sz, bytex *pnum);
extern unit numcvtint(intx sz, bytex *pnum) ;
extern intx numcvtchr(intx sz, intx exp, bytex *pnum, bytex *p) ;
extern intx chrcvtnum(bytex *p, intx len, bytex *pnum, intx *exp) ;
extern intx expcvtnum(bytex *str, intx len, bytex *pnum, intx *rexp) ;
extern unit stncvtlong(bytex *p);
extern intx stncvtint(bytex *p);
extern doublex expcvtfloat(bytex *str);
extern intx compare_bytex(bytex *ndp, intx snd, bytex *cdp, intx scd);
extern intx compare_number(bytex *ndp, intx snd, bytex *cdp, intx scd) ;
extern intx compare_qname(bytex *ndp, intx snd, bytex *cdp, intx scd, intx wc_t) ;
extern intx convert_bytex2(struct Session_ *so, intx ndt, intx xt, intx sz, bytex *ndp, bytex *tp);
extern intx convert_bytex(struct Session_ *so, intx ndt, bytex *np, bytex *p);
extern intx convert_number2(struct Session_ *so, intx ndt, intx xt, intx size, bytex *ndp, bytex *pnum);
extern intx convert_number(struct Session_ *so, intx ndt, bytex *np, bytex *pnum);
extern intx compare_bi(struct Session_ *so, bytex *np, intx nlen, bytex *cp, intx clen);
extern intx compare_nd_ind(struct Session_ *so, bytex *np, bytex *cp);
extern intx compare_nd(struct Session_ *so, bytex *np, bytex *cp);
extern void copy_bytex(bytex *s, bytex *d, intx n);
#ifdef __cplusplus
}
#endif

#endif
