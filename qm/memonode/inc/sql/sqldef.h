#ifndef _H_SQDEF
#define _H_SQDEF

#include <stdarg.h>
#include "misc/vms.h"
#include "misc/expt.h"
#include "qml/inf.h"
#include "prm/pi_cd.h"
#include "sql/sqlercd.h"

//pibd.h, sqlcb.h순으로 astnType의 인덱스 값을 맞춘다.
//application에서 C로 사용할수있게 클래스타입은 여기 위치하지않는다.
#define SQL_AST_T_IF_LOOP		14
#define	SQL_AST_T_COLUMN		15	//AST_T_VARARG_FUNC에 이어받기
#define SQL_AST_T_MAKE_VECTOR	16	
#define SQL_AST_T_IN			17
#define SQL_AST_T_NOT			18
#define SQL_AST_T_EXIST			19
#define SQL_AST_T_IS_NULL		20 
#define SQL_AST_T_MAKE_ISLOT	21	
#define	SQL_AST_T_SELECT		22
#define SQL_AST_T_BEGMARK		23	
#define SQL_AST_T_BETWEEN		24
#define	SQL_AST_T_PUSH_STACK	25	
#define SQL_AST_T_LATE			26	
#define SQL_AST_T_COLLECTION	27
#define SQL_AST_T_NONE			28
#define SQL_AST_T_SEQUENCE		29	//SQL_AST_T_SMAT2MVECT - depricate
									//여기까지는 calcStepAlloc에서 호출되므로 연속해야 한다.
#define	SQL_AST_T_INSERT		30
#define	SQL_AST_T_UPDATE		31
#define SQL_AST_T_DELETE		32
#define SQL_AST_T_EXECUTE		33	//k108so.
#define SQL_AST_T_EXECDDL		34
#define	SQL_AST_T_DESC_TAB		35
#define SQL_AST_T_COMMIT		36
#define SQL_AST_T_ROLLBACK		37	
#define SQL_AST_T_MARK_WP		38
#define SQL_AST_T_RETRO_WP		39
#define SQL_AST_T_VIEW			40
									//여기까지는 buildSql에서 호출되므로 연속해야 한다.
#define SQL_AST_T_BEG_AS		41
#define	SQL_AST_T_ASNAME		42	
#define	SQL_AST_T_TABLE			43
#define	SQL_AST_T_VALUES		44
#define SQL_AST_T_SELCOLS		45
#define SQL_AST_T_INTO			46	//이 밑에서 삽입 테이블 명시
#define SQL_AST_T_SET			47
#define SQL_AST_T_DELETE_TAB	48
#define SQL_AST_T_FROM			49
#define SQL_AST_T_WHERE			50
#define SQL_AST_T_ORDBY			51
#define SQL_AST_T_GRPBY			52
#define SQL_AST_T_HAVING		53
#define SQL_AST_T_ROLLUP		54
#define SQL_AST_T_CUBE			55
#define SQL_AST_T_FOR_UPDATE	56
#define SQL_AST_T_PAD_CLAUSE	57

#define VECTX_TP			17	//20	//dbi.h의 COLUMN_T_REFERENCE 타입을 이어서 설정된 값
#define SELMTX_TP			18	//21	
#define JMATX_TP			19
#define COLX_TP				20	//22	//COLUMN 상수
#define END_PACK_TP		COLX_TP	//단순히 JMATX_TP 다음 타입을 설정하여 리졸브측에
									//튜플 팩킹의 끝을 알린다.
#define NULL_PACK_TP	21
#define OVER_PACK_TP	22
#define ADDRESSING_TP	23

#define XTByteRange(tp) tp == POINTX_TP || tp == POINT2X_TP || tp == ILOBX_TP

#define REL_OP_TRUE		6	//REL_OP_LET에 이어서 설정, 비교시 무조검 참(크로스 조인을 위해)

#define J_INNER			0
#define J_LEFT_OUTER	1
#define J_RIGHT_OUTER	2
#define J_FULL_OUTER	3
#define IsLeftOuter(jop)	jop & 1
#define IsRightOuter(jop)	jop & 2
#define IsOuter(jop)		jop != 0

//내장함수 - 그룹 집계함수(0 ~ 4), SISQL_FUNC_COUNT이후와 일치해야함, sqlFuncSpec와 일치해야함
#define SQL_FUNC_COUNT		0
#define SQL_FUNC_MAX		1
#define SQL_FUNC_MIN		2
#define SQL_FUNC_SUM		3
#define SQL_FUNC_AVG		4
#define IsGrpFuncOp(fop) fop >= SQL_FUNC_COUNT && fop < SQL_FUNC_INITCAP //k371so_18.
#define SQL_FUNC_INITCAP	5
#define SQL_FUNC_LOWER		6
#define SQL_FUNC_UPPER		7
#define SQL_FUNC_LENGTH		8
#define SQL_FUNC_LENGTHB	9
#define SQL_FUNC_CONCAT		10
#define SQL_FUNC_SUBSTR		11
#define SQL_FUNC_INSTR		12
#define SQL_FUNC_LPAD		13
#define SQL_FUNC_RPAD		14
#define SQL_FUNC_LTRIM		15
#define SQL_FUNC_RTRIM		16
#define SQL_FUNC_ROUND		17
#define SQL_FUNC_TRUNC		18
#define SQL_FUNC_MOD		19
#define SQL_FUNC_CEIL		20
#define SQL_FUNC_FLOOR		21
#define SQL_FUNC_SYSDATE	22
#define SQL_FUNC_TOCHAR		23
#define SQL_FUNC_TONUMBER	24
#define SQL_FUNC_TODATE		25
#define SQL_FUNC_NVL		26
#define SQL_FUNC_ABS		27
#define SQL_FUNC_ROWNUM		28
#define SQL_FUNC_ADD_MONTH	29
#define SQL_FUNC_ROWID		30
#define SQL_FUNC_LEVEL		31
//sql함수의 컬럼옵셋 매개변수의 값으로서 이 두값과 컬럼옵셋이 아니면, 즉 0이면 스칼라 값을 나타내어
//런타임에 스택으로부터 스칼라값을 취한다.(이 두값은 리졸브컬럼에서 리턴되는 컬럼옵셋 값의 범위를 
//피하는 값으로 설정됐음)
#define SQL_CONST_INPUT		0xfffd
#define SQL_OPER_WC			0xfffe	//와일드카드
#define SQL_OPER_INPUT		0xffff	//스택에서 튜플을 팝하여 오퍼랜드로 사용하라는 의미 - depricate
#define VERSION_TAG	"version3.5.4" //slow_point.	current - k441so.9. last - k512so.
struct JoinTuple_;
//여기 Vector매크로들은 실렉트 매트릭스/튜플 구조를 실행 중간과정에서 컬럼 1짜리 1차원 백터로
//사용하기위한 매크로이다.
//여기 구조체의 동적 할당은 하나의 페이지내에서 할당하고 일괄 해제한다.
typedef struct SelTuple_ {
	sytex stType;//k181so.서버쪽만 정의한다.
	sytex fullPartNoCount;//k365so.mproc에서 첫번째 파트 헤드 셀퓨블만 의미있음
	bytex **pRowData;//실렉트 컬럼튜플 슬롯, 슬롯의 길이는 실렉트절의 실랙트컬럼 갯수만큼
					//단순 실렉트 컬럼이면 DataRecord상의 해당컬럼 옵셋이 가리키는 포인터 혹은
					//서브쿼리의 SelTuple의 pRowData의 해당컬럼 번째의 bytex *포인터이고
					//계산필드 컬럼이면 별도 할당된 포인터
	intx *szRowData;//레코드의 각 컬럼 데이터 길이 설정, 이값이 0이면 널값, 문자형이면 널이 
					//포함된 길이
	struct JoinTuple_ *jtupSor;
	struct SelTuple_ *ptrLeft, *ptrRight;//실렉트 컬럼 튜플을 연결하는 수직 포인터
//#ifdef MPROC_VER
	struct SelTuple_ *ptrNext;
//#endif
} SelTuple;
#define STupCData(srow, icol) *(srow + icol)
#define PointSTupColData(stup, icol) STupCData(stup->pRowData, icol)
#define LengthSTupColData(stup, icol) STupCData(stup->szRowData, icol)

#define TOP_SELECT_QUERY	-1
#define SUB_FROM_QUERY		-2 //실레트문의 from서브쿼리, insert, update문의 데이터 입력절 서브쿼리에 사용한다. 설령 insert/update문의 데이터 입력절에
							//계산식이 오고 그안에 서브쿼리가 있어서 서브쿼리 출력을 셀매트릭스로 해도 dml데이터 절의 계산식은 릴레이션이 있을수
							//없으므로 vect func로 처리되고 vect func는 오퍼랜드 타입으로 셀튜플도 취급하므로 상관없다.
#define SUB_WHERE_QUERY		-3
#define SUB_SELECT_QUERY	-4
#define SUB_SELECT_DIRECT	-5
#define DmlOrTopQuery(qry_t) qry_t > SUB_FROM_QUERY
#define TopOrFromQuery(qry_t) qry_t > SUB_WHERE_QUERY
#define NTopOrFromQuery(qry_t) qry_t < SUB_FROM_QUERY
#define NTopQuery(qry_t) qry_t < TOP_SELECT_QUERY
typedef struct {
	intx cntSelCol;//컬럼은 0부터 시작
	numx cntSRow;//실렉트 로우 갯수
	bytex *tSelCol;//COLUMN_T_SHORT~, 실렉트 컬럼 갯수만큼 각 실렉트 컬럼의 타입이 런타임에 
	//설정되는데 단순실렉트 컬럼이면 DQIMakeJoinToSelection, 계산필드이면 DQIAttatchSelection에서 설정된다.
	bytex **nmSelCol;//sql net에의해 결과를 리졸브할경우 sql state가 없으므로 여기 컬럼이름을 적재
	SelTuple *firstSelTup;//첫번째 실렉트 튜플 포인터
	void *selState;//현 셀 매트릭스를 생성하는 sql state
	shortx selRelSub;//주쿼리의 실렉트절의 연관서브쿼리에서 리턴된 셀메트릭스를 나타냄,
					//DQISelSubSelectControl에서 설정된다.
	shortx cvtsubvect;//clus버젼에서만 사용
} SelMatrix; 
#define TypeSColSMat(smat, icol) *(smat->tSelCol + icol)
#define NameSColSMat(smat, icol) *(smat->nmSelCol + icol)
#define TypeSColSMat2(qc, icol) TypeSColSMat(qc->resultSet, icol)
#define StatSMat(smat) ((SqlState *)smat->selState) //k364so.

#define InitSMatrix_(smat, sel_cnt) {\
	smat->cntSelCol = sel_cnt;\
	smat->cntSRow = 0;\
	smat->firstSelTup = nullx;\
	smat->nmSelCol = nullx;\
	smat->selRelSub = 0;\
}
#define InitSMatrix(imem, sel_cnt, smat) {\
	IMemAlloc(SelMatrix, imem, sizeof(SelMatrix), smat);\
	InitSMatrix_(smat, sel_cnt);\
	IMemAlloc(bytex, imem, sel_cnt, smat->tSelCol);\
}
typedef struct JoinTuple_ {
//#ifdef DCACHE_VER
	sytex stType;//k181so.
//#endif
	shortx avlFactor;//k181so.
	uintx linkLevel;//k364so.
	void **joinRow;//실제 테이블의 데이터레코드(DataRecord) or SelMatrix의 실렉트 튜플(SelTuple)
	void *supSorTup;//상위 테이블의 소스 튜플 
	struct JoinTuple_ *ptrLeft, *ptrRight;//필터된 레코드 튜플을 연결하는 수직 포인터
									//데이터레코드 포인터 슬롯의 길이는 from절의 테이블 갯수만큼
	struct JoinTuple_ *ptrNext;//#ifdef MPROC_VER
	struct JoinTuple_ *ptrNext2;//k337so.
} JoinTuple;
#define PEndGrpJTup(jtup) jtup->ptrNext2 //k337so.
#define PIJTup_(jrow, itab) *(jrow + itab)
#define PIJTub(jtup, itab) PIJTup_(jtup->joinRow, itab)
#define PTabJTub(jtup, itab) (DataRecord *)PIJTub(jtup, itab)
#define PSelJTub(jtup, itab) (SelTuple *)PIJTub(jtup, itab)
#define _CopyJoinTuple(tarjrow, sorjrow, ntab) {\
	void **ntar = tarjrow + ntab;\
	for(;tarjrow < ntar; sorjrow++, tarjrow++) *tarjrow = *sorjrow;\
}
#define CopyJoinTuple(tarjtup, sorjtup, ntab) {\
	void **tarjrow = tarjtup->joinRow;\
	void **sorjrow = sorjtup->joinRow;\
	_CopyJoinTuple(tarjrow, sorjrow, ntab);\
}
struct ByVector_;
#define FICTION_JOIN	2
typedef struct JoinMatrix_ {
	numx cntJRow;//죠인 로우 갯수, 조건절에서 참/거짓 판단을 조인매트릭스로 하므로 데이터가 없이
				//단순 참을 나타낼때는 -1이 설정된다. 실제 조인튜플 갯수를 알고싶을때는 
				//CntJoinRow_를 사용한다.
	shortx dirtyJoin;
	shortx cntJoinTab;
	void *sresJMat;
	void *grpvectJmat;
	bytex *indicJTab;
	bytex *indicSupRel;
	JoinTuple *firstJTup;//첫번째 죠인튜플 포인터
	JoinTuple *clusJTup;//#ifdef CLUS_VER, clus right outer join list
//#ifdef MPROC_VER
	intx cntPartLink;
	shortx combinedJMat;//밑의 것도 intx사이즈로 같이 리셋하므로 두개의 위치가 바뀌면 안된다.
	shortx devidePartList;
	void *eachProc;//메인프로세스를 제외한 병렬프로세스만 설정된다.
	struct JoinMatrix_ *supJmatMain;//k7so.
	struct JoinMatrix_ *ptrLeft, *ptrRight;
//#endif
	void *reentKeepGrpJm;//k371so_37.
} JoinMatrix;

#ifdef CLUS_VER
#define InitClusJList(jmat) jmat->clusJTup = nullx
#else 
#define InitClusJList(jmat)
#endif
#define AllocJMatrix_(qc, jmt, sres) {\
	AllocQCData(JoinMatrix, qc, sizeof(JoinMatrix), jmt);\
	jmt->cntJRow = jmt->dirtyJoin = 0;\
	jmt->grpvectJmat = nullx;\
	InitClusJList(jmt);\
	jmt->firstJTup = 0;\
	jmt->cntJoinTab = 0;\
	jmt->sresJMat = sres;\
}
#ifdef MPROC_VER
#define AllocJMatrix(qc, jmt, sres, jmsor) {\
	AllocJMatrix_(qc, jmt, sres);\
	jmt->eachProc = nullx;\
	jmt->cntPartLink = 0;\
	jmt->supJmatMain = nullx;\
	*(intx *)&jmt->combinedJMat = 0;/*jmt->devidePartList = 0;*/\
	NewIndicJMatrix(qc, sres->resultState->cntTable, jmt, jmsor);\
}
#define CopyJMatHead(t_jmat, s_jmat) {\
	t_jmat->cntJRow = s_jmat->cntJRow;\
	t_jmat->cntJoinTab = s_jmat->cntJoinTab;\
	t_jmat->firstJTup = s_jmat->firstJTup;\
	t_jmat->dirtyJoin = s_jmat->dirtyJoin;\
	t_jmat->cntPartLink = s_jmat->cntPartLink;\
	t_jmat->devidePartList = s_jmat->devidePartList;/*k371so_4.*/\
}
#else
#define AllocJMatrix(qc, jmt, sres, jmsor) {\
	AllocJMatrix_(qc, jmt, sres);\
	NewIndicJMatrix(qc, sres->resultState->cntTable, jmt, jmsor);\
}
#define CopyJMatHead(t_jmat, s_jmat) {\
	t_jmat->cntJRow = s_jmat->cntJRow;\
	t_jmat->cntJoinTab = s_jmat->cntJoinTab;\
	t_jmat->firstJTup = s_jmat->firstJTup;\
	t_jmat->dirtyJoin = s_jmat->dirtyJoin;\
}
#endif
#define AllocJMatrix2(_imem, _jmt, _jmsor) {\
	IMemAlloc(JoinMatrix, _imem, sizeof(JoinMatrix), _jmt);\
	_jmt->sresJMat = _jmsor->sresJMat;\
	CopyJMatHead(_jmt, _jmsor);\
	NewIndicJMatrix2(_imem, SStateJmat(_jmt)->cntTable, _jmt, _jmsor);\
	_jmt->supJmatMain = nullx;\
}
#define CountIndicJMat(_jmat, cnt) for(cnt = 0;*(_jmat->indicJTab + cnt); cnt++) //k364so.
#define ResetIndicJMat(jmat, _itab) *(jmat->indicJTab + _itab) = 0
#define SetJMatrix__(jmat, _itab) {\
	if(*(jmat->indicJTab + _itab) == 0) {\
		*(jmat->indicJTab + _itab) = 1;\
		jmat->cntJoinTab++;\
	}\
}
#define SetJMatrix_(jmat, njtup, itab_1, itab_2) {\
	jmat->cntJRow = njtup;\
	if(itab_1 >= 0) {\
		SetJMatrix__(jmat, itab_1);\
		jmat->dirtyJoin = 1;\
	}\
	if(itab_2 >= 0) {\
		SetJMatrix__(jmat, itab_2);\
		jmat->dirtyJoin = 1;\
	}\
}
#define SetJMatrix(jmat, njtup, htup, itab_1, itab_2) {\
	SetJMatrix_(jmat, njtup, itab_1, itab_2);\
	jmat->firstJTup = htup;\
	CHECK_COMBINE_JMAT(jmat);\
}
#define ResetJMatrix(jmat, ntab) memset(jmat->indicJTab, 0x00, ntab)
//이전 조인이 있는데 기조 조인된 결과가 없으면 1리턴
#define IsEmptyJoin(jmat) jmat->dirtyJoin && jmat->cntJRow == 0

#define JMatCurRelRow(jmat) SresCurRelRow((SResJmat(jmat)))
#define JMatSupRelRow(jmat) SresSupRelRow((SResJmat(jmat)))
#define JMatSupRelIdx(jmat) (SResJmat(jmat))->itabRelSup

#define IsCrossJoin(jmat) jmat->cntJRow > 0
#define IsJoinTable___(indic, itab) *(indic + itab)
#define IsJoinTable__(jmat, itab) IsJoinTable___(jmat->indicJTab, itab)
#define IsJoinTable_(jmat, rc) IsJoinTable__(jmat, rc->idxTable) //rc가 본컬럼(상위가아닌)일때 사용
#define JmatSupRel__(indic, itab) *(indic + itab)
#define JmatSupRel_(jmat, itab) JmatSupRel__(jmat->indicSupRel, itab)
#define JmatSupRel(jmat, rc) JmatSupRel_(jmat, rc->idxTable)
#define ResetIndicSupRel(jmat, ntab) memset(jmat->indicSupRel, 0x00, ntab)
#define AllsetIndicSupRel(jmat, ntab) memset(jmat->indicSupRel, 0x01, ntab)
#define CopyIndicJMat2SupRel(jmat, ntab) memcpy(jmat->indicSupRel, jmat->indicJTab, ntab)

#define NewIndicJMatrix_(_ntab, _jmat, _jmsor) {\
	register intx _itab;\
	register bytex *_p, *_p2, *_psor, *_psor2;\
	if(_jmsor) {\
		for(_itab = 0, _p = _jmat->indicJTab, _psor = _jmsor->indicJTab, \
			_p2 = _jmat->indicSupRel, _psor2 = _jmsor->indicSupRel; _itab < _ntab; _itab++) {\
				*(_p + _itab) = *(_psor + _itab);\
				*(_p2 + _itab) = *(_psor2 + _itab);\
		}\
	} else {\
		for(_itab = 0, _p = _jmat->indicJTab, _p2 = _jmat->indicSupRel; _itab < _ntab; _itab++) {\
			*(_p + _itab) = 0;\
			*(_p2 + _itab) = 0;\
		}\
	}\
}
#define NewIndicJMatrix(_qc, _ntab, _jmat, _jmsor) {\
	register intx __ntab = _ntab;\
	AllocQCData(bytex, _qc, __ntab, _jmat->indicJTab);\
	AllocQCData(bytex, _qc, __ntab, _jmat->indicSupRel);\
	NewIndicJMatrix_(__ntab, _jmat, _jmsor);\
}
#define NewIndicJMatrix2(_imem, _ntab, _jmat, _jmsor) {\
	register intx __ntab = _ntab;\
	IMemAlloc(bytex, _imem, __ntab, _jmat->indicJTab);\
	IMemAlloc(bytex, _imem, __ntab, _jmat->indicSupRel);\
	NewIndicJMatrix_(__ntab, _jmat, _jmsor);\
}
/****************** Code Page structure begin - 코드페이지 해제시 일괄해제 *******************/
#define SENSELESS_COL_IDX	32767	//참조 컬럼 갯수가 이 정도 값이 될리없다.
#define SENSELESS_TAB_IDX	127	//from절 테이블 갯수가 이 정도 값이 될리없다.
//아래 두개 자료구조의 초기화는 SqlState로드시 초기화될때 같이 된다.
//아래 두개 자료구조에서 컬럼이름들은 SqlBlobHead로 할당되고 옵셋주소는 해드를 가리킨다.
typedef struct {
	ushortx siSqlType;
	ushortx regSqlStat;//조건절의 서브쿼리에서 상위 컬럼을 참조할경우 상위 sql result/state(그 상위컬럼이 
				//보관되는)가 저장되는 레지스터 인덱스를 나타내고 이때 idxColumn, idxTable은 
				//상위 sql state내의 테이블인덱스및 그 테이블의 컬럼인덱스를 나탄내다. 
				//자신쿼리의 테이블(서브쿼리포함)을 참조하는 컬럼은 모두 자신의 인덱스를 나타낸다.
				//직접 subState의 주소를 적재하지않고 이를 적재하는 레지스터옵셋으로
				//하는 이유는 SqlResult의 selMatrix는 데이터로서 코드와 데이터가 분리되어 같이
				//움직여야 하기때문
	intx nmRealCol;//SqlBlobHead의 옵셋주소(블랍이 테이블이름.컬럼이름이면 컬럼이름만)
	intx iSqlStat;
	shortx idxColumn;//컬럼은 0부터 시작
	sytex idxTable;//SqlState의 sql spec index를 표시, orderby절에서 실렉트컬럼인덱스가 설정되어
					//이 인덱스로서 컬럼리졸브되면 이 테이블 인덱스값은 -1로 설정됨
	sytex descOrd;// - depricate
	sytex superRef;//이 값이 1 이면 regSqlStat가 상위 sql state를 가리킨다는 의미, 이 참조컬럼은 
	//상위 문장의 테이블을 참조 한다는 의미, 0이면 regSqlStat가 자신 sql state를 가리킨다는 의미,
	//즉 런타임에 이 값이 0인 참조컬럼이 죠인연산을 수행하면 그 타겟 죠인튜플은 죠인스택의 탑에 
	//있는 죠인튜플이 된다.(현행 조인 튜플이므로)
	sytex distinctCol;//집계함수의 매개변수로 컬럼이 올때 distinct가 명시되면 설정되고 실렉트절
	//의 distinct표시는 실렉트절 전체에 적용되는데 이는 sql state의 distSelect에 설정된다.
	sytex whereHaving;
	sytex pad;
	intx smatClusChain;
	bytex *nameRealCol;
#ifndef OPT_ADDR64
	void *pad2;
#endif
} ReferColumn;
#define IsCurrentTable(rc)	rc->superRef == 0
#define IsSuperTable(rc) rc->superRef
typedef struct {
	ushortx siSqlType;
	sytex tSelColumn;//COLUMN_T_SHORT ~ COLUMN_T_TIME, 계산필드가 아닌 컬럼은 load sql state에서
					//설정되고 계산필드컬럼은 DQIAttatchSelection에서 설정된다.
	sytex justSelCol;//k157so.
	bytex pad[4];
	intx resCol;//refer column offset address, 계산식 혹은 서브쿼리 실렉트 컬럼이면 0 설정
	intx nmFormCol;//SqlBlobHead의 옵셋주소, 별명이 지칭되면 별명으로 본명만 있으면 본명으로 설정
					//별명이 없으면 별명과 본명을 같게설정
	ReferColumn *resolveCol;
	bytex *nameFormCol;
#ifndef OPT_ADDR64
	void *pad3[2];
#endif
} SelectColumn;
#define JustSelCol(sc) sc->justSelCol	//k157so.참조컬럼 옵셋이 설정되있으면 단순(계산식 혹은 서브쿼리가 아닌)
									//실렉트 컬럼이다.
#define CalcSelCol(sc) sc->justSelCol == 0 //k157so.
/****************** Code Page structure end *************************************/
#include "misc/stack.h"
//k45so.prec은 update에서 변경전후 레코드 사이즈가 같을경우 trs tagging하기위해 복사된 포인트일수있다.즉 러닝레코드 포인트가 아닐수있음
typedef void (*FPSyncCallBack)(sytex trs_code, seg_t_adr arec, bytex *prec);

#define TP_JTRS_OPEN		0
#define TP_JTRS_CLOSE		1
#define TP_JTRS_BLOB		2
#define TP_JTRS_SQL			3
#define TP_JTRS_OVER		4

typedef struct {
	shortx typeJrnTrs;
	shortx oprJrnTrs;
	intx jrnUserId;
	uintx szJrnData;//pure data size(except head size)
	bytex *pJrnData;
} JournalHead;

struct InstantMem_;
typedef void *tJInst;
typedef void *tASync;
typedef void *tSync;
typedef void *tSqlCb;
typedef void *tDbe;
typedef void *tQns;
typedef void *tDbi;
typedef void *tSdb;
typedef void *tMdb;
typedef void *tPdb;
typedef void *tNdb;
typedef void *tJrnMgr;
typedef struct {
	intx szDSlot, szLSlot;
	intx icolMSel;
	SelMatrix *smatMan;
} SMatMaker;
struct SocketAlarmBoard_;
typedef struct SocketAlarm_ {
	shortx initWaitCount;
	shortx returnWaitCount;
	shortx forceCloseQrySoc;
	intx qcsocfd;
	struct SocketAlarmBoard_ *socAlarmBoard;
	struct SocketAlarm_ *ptrLeft, *ptrRight;
} SocketAlarm;
typedef struct SocketAlarmBoard_ {
	SocketAlarm *lsocAlarm;
	hmutex mutSocAlarm;
	volatile shortx *netAliveDisplay;
} SocketAlarmBoard;
typedef struct PreRangePart_ {
	numx cntPageRange;
	seg_t_adr begPageRange, endPageRange;
	JoinTuple *partjtup;
	struct PreRangePart_ *ptrLeft, *ptrRight;
} PreRangePart;
typedef struct SelectPartTag_ {
	intx lenpartname;
	bytex *selpartname;
	seg_t_adr selpartaddr;
	struct SelectPartTag_ *ptrLeft, *ptrRight;
} SelectPartTag;
#define EXT_BLOB_SIZE	0x8000000	//128M
//#include "misc/fio.h"
typedef struct {
	uintx sizeBlob;
    uintx remainBlob;
    uintx offBlobPage;
	uintx totalOffset; //mnj@3. jni_ver
	uintx overWriteLen; //mnj@3. jni_ver
    bytex *pageBlob;
} BlobStruc;
struct SqlQuery_;
typedef struct {
	//intx freeQCRes;//이값이 1이면 imemQuery 해제하고 리턴
	sytex monitorMode;//최종 실렉션에서 모니터 모드이고 고정길이이면 포인터 assign하고 그외는 모두 
	//복사한다. 복사를 하는 이유는 실렉션과정까지만 (스케쥴)뮤택블럭 보호되고 그 이후 사용자에의한
	//패치과정은 보호되지않으므로 패치과정중 변경이 발생기때문, 모니터모드이면 뮤택블럭API에의해 
	//보호블럭안에서 패치를 하므로 가변길이 테이블 실렉션도 복사를 안해도 될거라고 생각되지만 이것이
	//아닌이유는 모니터모드는 한번 실렉션한 것을 여러번 반복하여(뮤택설정,해제를 반복)사용하므로 
	//가변길이 테이블을 패치한 것은 데이터자체가 깨질수있으므로 복사를 해야한다.
	sytex idCursor;
	sytex jumpRelSequence;
	sytex suprelcmpjoin;//k360_34so.
	tJInst jobControl;//JobInstance
	SelMatrix *resultSet;//imemQuery에 의하여 할당, imemQuery해제에의한 일괄해제
						//sqlDecoder에서 실행 시작 시점에 전 실행의 셋이 리셋되므로 따로 해제않는다.
	InstructStack *sqlIpStack;
	UnitStack *oprStack, *joinStack;//DirectQuery 혹은 sqlprs에서 SqlQuery의 스위치 파라메터로
									//설정되어야 한다.
	bytex *sqlCodeBase;//파싱방식 - celldef, 직접호출 - 파이프코더의 cmemBuild의 savepage
	bytex *sqlRegBase;//파싱방식 - cell context의 regVarSlot, 직접호출 - 다이렉트쿼리의 registerSlot
	void *imemQCData;
	void *imemQuery;//join tuple등의 생성, 파싱방식 - 실렉트 로직에서 생성,
									//직접호출 - 파이프코더의 imemBuild
	void *imemVect;//중간 처리과정 생성용, 내부에서 라이프사이클이 끝나면 그때그때 해제
									//생성과 해제 시점은 imem query를 따라간다.
	void *topSelSres;//k461so.실레트 튜플 패치할때 컬럼정보등을 알기위해 설정됨
	//이하 패치과정에서만 필요한 할당에 관한 멤버들을 따로 분리하여 패치핸들로 묶고 사용자가 이 핸들을
	//직접 해제하기전까지는 패치핸들에서 할당된 패치데이터들을 계속사용할수있게하면 좋으나 이렇게 할수
	//없는 이유는 실렉트매트릭스를 생성할때 조인튜플에서 새로할당복사(이렇게하는 이유는 패치하는 중에
	//실렉트된 데이터가 변경될수있으므로)하고 이 복사된 실렉션을 fetch API에서 바로 포인터로 리턴시키기
	//때문에 패치과정과 실렉션과정이 분리될수없기 때문이다.
	intx sfdSqlnet;
	intx sfdSrvnet;
	void *imemBwdCycle;//k371so_17.begin출력전 후방 순환재실행 단계동안 유지되는 메모리 할당을 위해 사용됨, 전방 출력후엔 리셋됨.
	void *reentKeepGrp[2];
	shortx notFreeBwd;//end
	sytex togReentKeep;//k371so_35.begin
	sytex normNotGrpKeep;//end.
	intx cntReentCycle;//k423so.
	sytex andOrReentSeq;//k371so_38.
	bytex pad[57];
	//intx cntQCCumAlc, cntQCNewAlc, cntQCGetAlc, cntQCPool;
	bytex *qcAlcPoint;//이하 4개 변수는 셀매트릭스 패치 과정에서 할당에 사용되는 변수
	void *qcHeadPage;//셀매트릭스 패치 과정에서 할당된 페이지 리스트
	void *qcCurPage;//현재 할당된 페이지
	void *qcPagePool;//리셋된 페이지 리스트
	intx icolCurStup;
	SelTuple begFetchStup, *curFetchStup;
	bytex *dateFmt2c, *dTimeFmt2c, *timeFmt2c, *digitFmt2c;
	bytex *dateFmt2d, *dTimeFmt2d, *timeFmt2d;
	sytex remainFetch, remainFetch2;//이값이 1이면 계속 패치한다.(이값이 0가 될때가지 패치), 나중에 remainFetch2관련 제거
	sytex extBlobData;//k361so.
	sytex finClusRelate;//k446so.
	//shortx preventRemFetch;
	//intx cntDevideFetch, szSqlCxtStack;
	//unit sqlCxtFrame[RST_FRM_SIZE];
	//UnitStack lSqlCxtStack, *sqlCxtStack;
	//InstructStack IpRowPrs, *ipRowPrs;
	//PrimeInstruction *ipRowPrsFrame[RST_FRM_SIZE];
	intx gridDocking;
	unit extBlobId;//k361so.
	tDbi focusBlobDbi;
	void *focusBlobTbi;
	void *focusBlobRec;
	seg_t_adr aHeadBlob;
	intx cntBlobRdc, offsetBlob;
	void *pCurBlob;
	intx sqErCode;//이 값이 0이면 성공, 음수이면 에러
	bytex sqErMsg[ERRMSG_SZ];//첫번째 바이트는 에러여부 표시(-1이면 에러), 두번째 바이트부터 에러 메세지
	SMatMaker smatMaker;
	intx dynComLen;//k520so.1.
	intx onJrnQry;
	void *manJrnQry;
	JournalHead jrnheadQry;
	void *qcHourFac;
	void *qcProcHold;//#ifdef MPROC_VER
	void *procBinder;//
	void *qcSession;//#ifdef CLUS_VER, 이값이 있으면 먼저 리셋,해제한다.
	void *propaResTable, *propaResTabSP;//#endif, 테이블 스팩의 invisibleRTab 설명 참조
	shortx cvtBOrder;
	shortx szJrnSql;
	shortx jrnSqlType;
	ushortx hintQc;
	bytex *jrnSqlStmt;
	BlobStruc blobSerial;
	bytex *fastBuffer;
	SocketAlarm qcSocAlarm;
	bytex *dynComSlot;//k61so.
	struct SqlQuery_ *sqOwner;//k146so.
	void *proQC, *endInstr;//k198so.
	intx rowIdx;  //mnj@3. jni_ver
	intx elobOn;  //mnj@2. k361so.
	longx elobId; //mnj@2. k361so.
	intx elobCon; //mnj@4.
//#if defined(PART_VER)
	intx cntPreRange;
	numx cntTotPage;
	PreRangePart *lpreRanage;
	SelectPartTag *ldelPart;
//#endif
//#ifdef MPROC_VER
	void *comprocQc;//k376so_2.
//#endif
	void *fpReflection;//403so.
} QueryContext;

#define JobCntlQC(qc) (JobInstance *)qc->jobControl
#define JCLIdQC(qc) (JobCntlQC(qc))->idJobInst
#define RemoteSQLQuery_(qc) qc->sfdSqlnet
#define ErrorMsgQC(qc) &qc->sqErMsg[ORDNUM_LEN +1]
#define ResetErQC(qc) {\
	qc->sqErCode = 0;\
	qc->sqErMsg[ORDNUM_LEN] = 0;\
}
#define OccurQCReent(qc) qc->remainFetch
#define NOccurQCReent(qc) qc->remainFetch == 0
#ifdef REENT_VER
//소팅이나 그룹과 같이 집계 기능에서 tms_group과 job의 인덱스, segs를 사용하는데 그룹처리 다음 소팅이 또 있을 경우에
//한 sql에 위 객체들이 두개씩이 필요하고(세개는 필요없음. 실행이 2개까지만 중첩되기때문에) 순환재실행중에 한번 바인딩되면
//바인딩된 객체들이 계속사용되야하므로 슌환재실행 방향이 바뀌는 전후 시점에 본 매크로를 호출하여 toggle로서 그리 한다.
#define ToggleReentGrp(qc, normal_tog) {/*k371so_35.*/\
	if(normal_tog) {/*집계 기능이 아닌 곳에서의 호출은 방향이 바뀌는 시점에 한번만 토클설정되게하여 중복되는 경우에 처음 한번만 되게한다.*/\
		if(qc->normNotGrpKeep == 0) {\
			qc->normNotGrpKeep = 1;\
			qc->togReentKeep = !qc->togReentKeep;\
		}\
	} else {\
		qc->normNotGrpKeep = 0;\
		qc->togReentKeep = !qc->togReentKeep;\
	}\
}		
#define RESET_REENT_QC(qc, job, reent_close) {/*k371so_17.순환재실행동안 그룹연산을 위해 수행됐던 자원들을 해제한다.*/\
	qc->notFreeBwd = 0;\
	ResetReentPop(job);\
	if(qc->reentKeepGrp[qc->togReentKeep]) {\
		/*k399so.DeleteCIndex(((TMSProcGroup *)qc->reentKeepGrp)->reentGrpIndex, 0, 1);\
		CloseSegStore(((TMSProcGroup *)qc->reentKeepGrp)->reentGrpSegs, 0, 1);*/\
		((TimeSliceProceduer *)job->dbaseEngine->tmsProceduer)->putTMSGroup((TMSProcGroup *)qc->reentKeepGrp[qc->togReentKeep]);\
		qc->reentKeepGrp[qc->togReentKeep] = nullx;\
		/*k397so.job->vmLockMade = 0;/*k371so_17_3.*/\
	}\
	if(reent_close) {\
		if(qc->reentKeepGrp[!qc->togReentKeep]) {/*k371so_35.*/\
			((TimeSliceProceduer *)job->dbaseEngine->tmsProceduer)->putTMSGroup((TMSProcGroup *)qc->reentKeepGrp[!qc->togReentKeep]);\
			qc->reentKeepGrp[!qc->togReentKeep] = nullx;\
		}\
		qc->normNotGrpKeep = 0;\
		/*k371so_37_2.for(TMSProcGroup *tpg = job->lTmsGrpReent;tpg; tpg = tpg->ptrRight) job->dbaseEngine->tmsProceduer->putTMSGroup(tpg);/*k371so_37.*/\
		/*k371so_37_2.if(job->imemListReent) CAT_LIST(InstantMem, job->imemListMProc, job->imemListReent);/*k371so_37.*/\
	}\
}
#else 
#define RESET_REENT_QC(qc, job, reent_close)
#endif
#define IsItSocAlarm(qc) qc->qcSocAlarm.socAlarmBoard && qc->qcSocAlarm.socAlarmBoard->netAliveDisplay == nullx	//k58so.
#define SelLength(stup, icol) LengthSTupColData(stup, icol)
#define SelPoint(stup, icol) PointSTupColData(stup, icol)
#define SelLength2(qc, icol) SelLength(qc->curFetchStup, icol)
//마지막 명령이 실렉트문이어서 셀 매트릭스가 리턴되었으면 커런트를 초기 설정한다.
//로우가 1개인 패치는 FINextSelection을 호출하지않아도 패치할수있게 설정한다.
#define SetCurResSet(qc) {\
	qc->curFetchStup = &qc->begFetchStup;\
	if(qc->resultSet && qc->resultSet->cntSRow) {\
		qc->curFetchStup->pRowData = qc->resultSet->firstSelTup->pRowData;\
		qc->curFetchStup->szRowData = qc->resultSet->firstSelTup->szRowData;\
		qc->curFetchStup->ptrRight = qc->resultSet->firstSelTup;\
	} else {\
		qc->curFetchStup->pRowData = nullx;\
		qc->curFetchStup->szRowData = nullx;\
		qc->curFetchStup->ptrRight = nullx;\
	}\
}
#define SetCurResMatrix(qc, sres) {\
	qc->resultSet = sres->resultMatrix;\
	SetCurResSet(qc);\
}
#define IsForceCloseQrySoc(qc) qc->qcSocAlarm.forceCloseQrySoc
#define IsAttatchedSocAlarm(sq) sq->descQC.qcSocAlarm.socAlarmBoard

#define CALC_SIZE_WRITE_BLOB_(bs) BLOB_SZ - bs->remainBlob
#define CALC_SIZE_WRITE_BLOB(qc) CALC_SIZE_WRITE_BLOB_((&qc->blobSerial))

#define BEG_REG_NORMAL_SQL_ARG	2
#define BEG_PLUG_NORMAL_SQL_ARG	3

#define NEGANET_ERR_DIV	-1
#define POSINET_RET_DIV	-2
#define REENT_HEART_BIT	-3 //k423so.

#define NET_ERR_SET(qc, ecd) qc->sqErMsg[ORDNUM_LEN] = ecd
typedef struct {
	sytex charModeCS;
	bytex *pWrCurs, *pWrBeg, *pWrEnd;
	bytex *csTypes;
	intx nCaSeq;
	intx nSequenceCS;
	intx iSeqCurs;
	void *puserd;
} CaSchemCurs;//k526so.2.
#define SZ_ITEM_MAX	128
typedef struct SqlQuery_ {
	QueryContext descQC;//이 안의 imemQuery는 스케줄러에서 mdb인스턴스 로직에의하여 실렉트 문이면 
	//설정된다. 이의 해제도 컴파일된 mdb언어 해제코드에의해 스케줄러에 의해 요청되어 PutInstMem에의해 
	//jobControl로 자원리턴됨, 이 포인터가 설정되있으면 내부에서 리셋된후 재사용된다.(이전 데이터는
	//못씀) 따라서 맨마지막 이 sql desc의 사용을 끝낼때 reset sql desc를 호출하여 해제한다.
	//이렇게하는 이유는 스케쥴러의 쓰레드와 사용자 쓰레드가 틀려서 어느 한쪽에서만 잡인스턴스로
	//로 자원요청을 해야하기때문, 해제는 시작코드에서 매개변수로부터 해제여부를 묻게하면 된다
	SwitchParameter switchParam;//sql요청할때 매개변수는 스위치파라메터, 쿼리컨택스트, .. 순서로 
	//정렬한다. 포인터 타입의 매개변수는 파이프언어에서의 호출과 일관성을 위해 더블 사이즈로 정렬한다.
	tJSche sqlScheduler;
	tJInst sqlJInst;
	tSqlCb sqlBuilder;
	sytex firstSqlTypePrepStmt;
	sytex linkFromPassive;//k146so.
	shortx offRegPrepStmt;
	shortx execQuery;
	shortx initQuery;
	shortx iGateway;
	shortx iGateCard;
	shortx twinSqlExec;
	shortx resessJinst;//k60so.depricate
	intx bindJobid;
	uintx stampBindid;//k57so.
	void *sqlDbi;
	void *sqlProcess;
	void *sqlPlug;
	CaSchemCurs *caSchemCurs;//k526so.2.
	struct SqlQuery_ *sqGateway, *ptrLeft, *ptrRight;
} SqlQuery;//서브쿼리를 고려하여 이 자료구조를 프로시듀어의 매개변수로 넘긴다.
#define RemoteSQLQuery(sq) RemoteSQLQuery_((&sq->descQC))
#define JobSQLQuery(sq) ((JobInstance *)sq->sqlJInst)
#define JobIdSQLQuery(sq) JobSQLQuery(sq)->idJobInst

#define EXEC_TYPE_SQL	1
#define EXEC_TYPE_PREP	2

#define CHK_FAULT_SEL
#ifdef CHK_FAULT_SEL
#define ChkFaultSelection(qc, icol) {\
	if(qc->curFetchStup == nullx || qc->curFetchStup->szRowData == nullx) icol = -1;\
	if(icol < 0 || icol >= qc->resultSet->cntSelCol) icol = -1;\
}
#define ChkFaultSelection2(qc, icol) {\
	if(qc->resultSet == nullx || icol < 0 || icol >= qc->resultSet->cntSelCol) return -1;\
}
#define ChkFaultSelection3(qc) if(qc->resultSet == nullx) return -1
#define ChkFaultSelection4(qc) if(qc->curFetchStup == nullx) return nullx
#define ChkFaultSmat(smat, icol) smat == nullx || icol < 0 || icol >= smat->cntSelCol
#define ChkFaultSelection5(smat, icol) {\
	if(ChkFaultSmat(smat, icol)) return nullx;\
}
#define ChkFaultSelection6(qc, icol) {\
	if(qc->curFetchStup == nullx || qc->curFetchStup->szRowData == nullx) return nullx;\
	if(icol < 0 || icol >= qc->resultSet->cntSelCol) return nullx;\
}
#define ChkFaultSelection7(qc, icol) {\
	if(qc->resultSet == nullx || icol < 0 || icol >= qc->resultSet->cntSelCol) icol = -1;\
}
#else
#define ChkFaultSelection(qc, icol)
#define ChkFaultSelection2(qc, icol)
#define ChkFaultSelection3(qc)
#define ChkFaultSelection4(qc)
#define ChkFaultSelection5(smat, icol)
#define ChkFaultSelection6(qc, icol)
#define ChkFaultSelection7(qc, icol)
#endif

#define REG_PAGE_SIZE	4096	//256(서브쿼리 sql state 갯수) * 16(AllocDoubleReg) 면 충분하다.
typedef struct {//네이티브(C, Java)에서 mdb를 direct access하기위한 구조
	tSqlCb dqiBuilder;//join tuple등의 생성은 sql builder ( pipe builder ) -> imemBuild로 한다.
	unit OprStFrame[RST_FRM_SIZE], JoinStFrame[RST_FRM_SIZE];
	bytex registerSlot[REG_PAGE_SIZE];//k227so.정수형이 적재될수있으므로 위 유닛 사이즈 정렬 뒤에 위치시킨다.
	UnitStack operandStack;
	UnitStack joinTupleStack;
	QueryContext directQC;
	shortx initDQI;
	shortx negateGetSch;
	shortx newStmtHandle;
	SqlQuery *sqDirect;
} DirectQuery;
#define JobDirQuery(dique) (JobInstance *)dique->directQC.jobControl
#define DQIAllocData(tp, dqi, sz, pdat) AllocQCData(tp, (&dqi->directQC), sz, pdat)
//uni type은 unit
#define Point2Uni(pv)			(unit)pv
#define Uni2Point(tp, uv)		(tp *)uv
#define Long2Uni(lv)			lv
#define Uni2Long(uv)			uv
#define UniValPoint(tp, uv)		(tp *)&uv
#define Double2Uni(dv)			*UniValPoint(unit, dv)
#define Uni2Double(uv)			*UniValPoint(doublex, uv)
#define Uni2Float(uv)			*UniValPoint(floatx, uv) //k368so.

//DQIRelColSca등과 같은 DQI API에서 상수를 매개변수로 넘겨줄때 사용하는 매크로
#define DQIDoubleArg(dval)	DOUBLEX_TP, Double2Uni(dval)
#define DQILongArg(lval)	LONGX_TP, Long2Uni(lval)
#define DQIStringArg(pstr)	POINTX_TP, Point2Uni(pstr)

struct RQueryIt_;
typedef void (*FPRearQuery)(QueryContext *qc, struct RQueryIt_ *rquery, bytex *sql_stmt);
typedef struct RQueryIt_ {
	intx sockRquery;
	bytex rConnectStr[128];
	void *handleRquery;
	FPRearQuery funcRquery;
	struct RQueryIt_ *ptrLeft, *ptrRight;
} RQueryIt;

typedef struct RelationKeeping_ {
	sytex tRegVal;
	unit relRCase;
	ushortx jid_;
//#ifdef DCACHE_VER
	shortx do_cprec_;
//#endif
	intx ntab_, crelCmd, devideRel, execDevideQry, vergin_rk;
	numx njtup_, cntRemJTup;
	intx tp_convert_, tp_reverse_, not_overwrite_;
	void *headjtup_, *headstup_, *sibrow_, *remainJTup, *jmat_, *sql_res, **p2RelRow;
	struct {
		intx itab_, tcol_, icol_, sdiv_;
		void *head, *cur, *mp_;
		void *rdc_, *rtb_, *wtb_;
//#if defined(DCACHE_VER) && defined(VMC_VER)
		void *vm_cxt;
//#endif
		unit cval_;
		void *sresSupRel;
		intx colSupJoin;
	} l, r;
	struct RelationKeeping_ *p2relkeep;
} RelationKeeping;

#define ALT_TAB_CREATE	0
#define ALT_TAB_ADD		1
#define ALT_TAB_MODIFY	2

#define L_DATE_EXE_2C		36	//포맷길이에서 날짜부분 년월일 3개바이트가 두바이트씩 늘어나고 널문자를 고려하여 +4
#define L_DTIME_EXE_2C		71	//포맷길이에서 년월일시분초 3개바이트가 두바이트씩 늘어나므로 널문자를 고려하여 +7
#define L_TIME_EXE_2C		36	//포맷길이에서 시간부분 시분초 3개바이트가 두바이트씩 늘어나므로 널문자를 고려하여 +4
#define L_DATE_FMT_2C		32
#define L_DTIME_FMT_2C		64
#define L_TIME_FMT_2C		32
#define L_DIGIT_FMT_2C		50
#define L_DATE_FMT_2D		32
#define L_DTIME_FMT_2D		64
#define L_TIME_FMT_2D		32

#define CVT_STR_INITCAP		0
#define CVT_STR_LOWER		1
#define CVT_STR_UPPER		2

#define NULL_POSOFF			0x80000000	//함수의 매개변수로 음수값도 올수있을 경우에
										//널값을 표시하기위해 사용
#define NULL_IDX_VAL		0x80000000
#define MAX_JOURNAL_BLOB	131072		//64k * 2
#define MAX_JOURNAL_SESSION	10
//SQIMarshalQuery를 하고난후 일반 매개변수 설정할때 베이스 + 파라미터 인덱스(1부터 시작)로 하기
//위해 베이스값 정의(인덱스에서 하나 적게 설정-왜냐하면 파라미터는 1부터 시작하므로)
#define BASE_SQL_PARAM_IDX	1

#define SYNC_EXP_DML_REFLECT_WORKING	1 //transact tagging은 하지않고 working상태만 있는 디비를 싱크하는 옵션
#define SYNC_EXP_DML_REFLECT_RUNNING	2 //저장이나 이중화 목적으로 transact tagging을 하는 디비를 싱크하는 옵션

#ifdef OPT_DLL_EXP
#define DLLAPI	WINAPI
#else 
#define DLLAPI
#endif

typedef void (*CSqlUserFunc)(void *param);

#endif
