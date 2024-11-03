
#include "misc/xt.h"

#ifdef BIGVM_VER	//k111so.
typedef unit	seg_t_fi;
typedef unit	bsizex;
#else
typedef uintx	seg_t_fi;
typedef uintx	bsizex;
#endif
typedef uintx	seg_t_po;
typedef unit	seg_t_adr;

#ifdef PAGE_HIDDEN	//-depricate(ProvisHead로 대신한다.)
#define FRAME_INDEX_OFF	sizeof(unit)	//페이지선두의 페이지인덱스
#define SEG_PAGE_LIMIT	0xFFFFE8	//SEG_PAGE_SIZE - FRAME_INDEX_OFF
#define PROB_FRM_INDEX(p, i) *(seg_t_fi *)p = i
#else
#define FRAME_INDEX_OFF	0
#define SEG_PAGE_LIMIT	SEG_PAGE_SIZE	//사용안됨
#define PROB_FRM_INDEX(p, i) 
#endif

#ifdef BIGVM_VER //k111so.
#define A_SEG_BLOCK		0x8000000000000	//k182so.4096 //2 peta byte(2의 48 * 주소사이즈[2의 3승]), vm frame에 사용되어 fully 2의 64승 주소(18 exa byte) 지원 사이즈
#define PROVIS_A_CLASS	0x7FFFFFFFFFFC8 //k182so.4040 	//A_SEG_BLOCK - 56
#define B_SEG_BLOCK		0x10000000	//256M	//임시사용 //나중에 진짜 메모리 기반 대용량 사용이면 0x800000000	//34G, 281(34G를 주소사이즈 8바이트로 나눈 갯수 4G에 페이지 사이즈 65k를 곱한 크기) tera 바이트 주소 지원, 
#define PROVIS_B_CLASS	0xFFFFFC8						//0x7FFFFFFC8	//B_SEG_BLOCK - 56
#define V_SEG_BLOCK		0x80000		//512k, k155so.
#define PROVIS_V_CLASS	0x7FFC8		//V_SEG_BLOCK - 56

#define C_SEG_BLOCK		0x10000		//65536, 65k
#define PROVIS_C_CLASS	0xFFC8		//C_SEG_BLOCK - sizeof(ProvisHead), MemDevice에서 할당한 페이지의 최종사이즈가 메모리 연결 링크를 포함하여 64k로 맞추기위해
#define CLUS_FRAME_INNER_SIZE		0xFFA0 //clus frame size에서 ClusPacket 사이즈(32바이트)와 정수형데이터 alignment over size(8바이트), 즉 40바이트를 감한 사이즈, cluspac.cpp alignment over size 설명 참조
#define REENT_SERIAL_INNER_SIZE		0xFFB0 //k371so.INST_PAGE_SIZE에서 ReentSerialPage 사이즈(16바이트)와 정수형데이터 alignment over size(8바이트), 즉 24바이트를 감한 사이즈
#ifdef MAXVM_VER
#define SIZE_REAL_MBLOCK 			A_SEG_BLOCK	//BIGVM_VER를 최대 사이즈로 적용
#else 
#define SIZE_REAL_MBLOCK 			B_SEG_BLOCK	//b class임시 설정으로 2TB 지원 //281 TB주소 지원
#endif
#define N_ICACHE_SLOT				60000 //v class size 종속
#else

#ifdef MINVM_VER //소형 장비용
#define A_SEG_BLOCK		0x8000000	//128M
#define PROVIS_A_CLASS	0x7FFFFC8	//A_SEG_BLOCK - 56
#else
#define A_SEG_BLOCK		0x10000000	//256M, vm frame에 사용되어 C class page size일경우 데이터베이스당 2TB할당까지 할당가능
									//256M는 2의 28승이고 주소사이즈 2의 3승을 빼면 2의 25승, 곱하기 페이지 크기 2의 16은 2의 41승 => 2TB
#define PROVIS_A_CLASS	0xFFFFFC8	//A_SEG_BLOCK - 56, 세그먼트 메모리에서 double사이즈 용량할당은 A class size의 프레임과 A class size 페이지로 한다.
#endif								
#define B_SEG_BLOCK		0x1000000	//16M
#define PROVIS_B_CLASS	0xFFFFC8	//B_SEG_BLOCK - 56
#define V_SEG_BLOCK		0x20000		//128k
#define PROVIS_V_CLASS	0x1FFC8		//V_SEG_BLOCK - 56
#ifdef OPT_PAGE64
#define C_SEG_BLOCK		0x20000		//131072, 128k
#define PROVIS_C_CLASS	0x1FFC8		//C_SEG_BLOCK - sizeof(ProvisHead), MemDevice에서 할당한 페이지의 최종사이즈가 메모리 연결 링크를 포함하여 128k로 맞추기위해
#define CLUS_FRAME_INNER_SIZE		0x1FFA0
#define REENT_SERIAL_INNER_SIZE		0x1FFB0 //k371so.
#else
#define C_SEG_BLOCK		0x10000		//65536, 65k
#define PROVIS_C_CLASS	0xFFC8		//C_SEG_BLOCK - sizeof(ProvisHead), MemDevice에서 할당한 페이지의 최종사이즈가 메모리 연결 링크를 포함하여 64k로 맞추기위해
#define CLUS_FRAME_INNER_SIZE		0xFFA0
#define REENT_SERIAL_INNER_SIZE		0xFFB0 //k371so.
#endif
#define SIZE_REAL_MBLOCK 			A_SEG_BLOCK
#define N_ICACHE_SLOT				15000 //v class size 종속
#endif

//전체 메모리 할당 사이즈인 SIZE_MAX_MEMALC는 SIZE_REAL_MBLOCK * N_REAL_MBLOCK이고 최대 한계 사이즈는 
//블럭단위 사이즈(SIZE_REAL_MBLOCK)에 페이지 사이즈를 곱한 크기로서 BIGVM_VER이 아니면 2TB, BIGVM_VER이면 281TB, MAXVM_VER이면 18 exa바이트 이므로 
//이를 넘지않게 설정한다. 
//N_REAL_MBLOCK은 메모리 블럭 사이즈가 256M로 했을때의 값이고 한계 체크를 메모리 할당에서 하지 않는것으로 바꿔 사용되지 않으므로 의미없으나 PRE_ALLOC_RMB때문에 놔둔다.
#ifdef EXPRESS_VER //k109so.
#define SIZE_MAX_MEMALC			0X40000000	//1G //0X20000000 //512M depricate
#define CHECK_POINT_WORK_FREE	0X20000000	//512M
#define INC_POINT_INDEX_FREE	0X20000000	//512M
#define N_REAL_MBLOCK			4	//이 값은 10진수이므로 핵사로 변환해야함
#elif OPT_ADDR64
#define SIZE_MAX_MEMALC			0X100000000	//depricate 4G 0X200000000	//8G 0x1E00000000 //128G 0x300000000 //12G , SIZE_REAL_MBLOCK이 이 값보다 크면 N_REAL_MBLOCK은 무시되고 SIZE_REAL_MBLOCK 사이즈까지 할당됨.
#define CHECK_POINT_WORK_FREE	0X80000000	//2G 0X100000000	//4G
#define INC_POINT_INDEX_FREE	0X40000000	//1G
#define N_REAL_MBLOCK			48 //32		//이 값은 10진수이므로 핵사로 변환해야함
#else
#define SIZE_MAX_MEMALC			0x90000000	//depricate 2.4G
#define CHECK_POINT_WORK_FREE	0X40000000	//1G
#define INC_POINT_INDEX_FREE	0X20000000 //500M
#define N_REAL_MBLOCK			9 //이 값은 10진수이므로 핵사로 변환해야함
#endif

#define SIZE_VM_MBLOCK 			0x1000000		//16M, 이 가상블럭 사이즈보다 할당사이즈가
												//크면 요구 할당사이즈로 블럭을 할당한다.
#define N_VM_MBLOCK 			15	//1

#define CACHE_MEM_SIZE			0x20000000 //512M //k182so.20000000
#define HASH_MEM_RATE			0.0001220703125 //임시 테스트를 위해 256M를 어드레싱 하기위해 SIZE_REAL_MBLOCK(256M)로부터의 축소비율 
													//256M * 0.0001220703125 = 32k, 32k / 8(주소사이즈) = 4096, 4096 * 65536(페이지 사이즈) = 256M

#define TRS_REFL_RECYCLE		0X8000000 //128M 0X40000000	//1G

#define COLLECT_TIME_GAP		1200	//second
#define MAX_SCH_AT	1000	//65535

#define BLOB_SZ		0X80000000	//2G

#define MAX_MEM_URATE 0.8 //k218so.

#define INST_PAGE_SIZE			PROVIS_C_CLASS
#define INST_PAGE_OUTER_SIZE	C_SEG_BLOCK	//provis head를 포함한 c class 사이즈
#define SZ_PACKING_OUTER	8	//packing과정에서 데이터유무(1바이트), 데이터길이 필드
							//사이즈(4바이트) 길이가 페이지사이즈에서 오버될수
							//있으므로 여유로 8바이트 추가
#define SZ_FRONT_PAC_OUTER		INST_PAGE_SIZE + SZ_PACKING_OUTER
#define SZ_FRONT_PAC_ORDNUM		SZ_FRONT_PAC_OUTER + ORDNUM_LEN
#define SZ_FRONT_PAC_ORDNUM2	SZ_FRONT_PAC_OUTER + ORDNUM_LEN2

#define DIFF_CIRCULAR_ULONG(bigger, smaller) (bigger >= smaller ? bigger - smaller : (UNSIGNED_LONG_MAX - smaller) + bigger)
#define DIFF_CIRCULAR_ULONG2(big, little, big2, little2, diff_big, diff_little, minus) {\
	if(big < big2) minus = 1;\
	else {\
		diff_big = big - big2;\
		if(diff_big == 0) {\
			if(little < little2) minus = 1;\
			else {\
				minus = 0;\
				diff_little = little - little2;\
			}\
		} else {\
			minus = 0;\
			if(little < little2) {\
				diff_little = little + (UNSIGNED_LONG_MAX - little2);\
				diff_big--;\
			} else diff_little = little - little2;\
		}\
	}\
}
//dbe.h
#define TMOUT_DEFAULT_NBLK_CONNECT	1
#define TMWAIT_FRONT_REQ_PASSIVE	10
#define PASSIVE_TM_TWIN_ESTAB	3 
#define RETRY_TM_TWIN_ESTAB		2
#define RETRY_CNT_TWIN_ESTAB	3
#define PASSIVE_TM_FRONT_REQ_ALLOW	2
#define TM_TWIN_CODY_FOR_LISTENER	2
#define CNT_CLOSE_MARK_TWIN_CONNECTION	0
#define CNT_ALIVE_CHECK_TWIN_CONNECTION	1

#define RET_TWIN_CONNECT_FAIL	-1
#define RET_TWIN_ALIVE_NO_ACTIVE	-2

#define TMWAIT_NO_ACTIVE_CONNECT	2
#define TMOUT_FORWARD_RETURN		2

#define TMOUT_RESESSION			60

#define PERMIT_CHANGE_FRONT_TMGAP	1 //k14so. //3 //2 //k11so. 3
#define TMWAIT_ACGATE_SHUTDOWN		10
#define CNT_CHECK_ACTIVE_REPORT		5 //k14so.
#define MSEC_TMOUT_ACTIVE_REPORT	100 //k14so.

#define TMWAIT_ACTIVE_CROSS_CONNECT	1

#define CNT_RETRY_ACTIVE_REPORT		4	//3회 반복

#define CNT_STANDALONE		3
#define TMWAIT_STANDALONE	3

#define SIZE_TRS_LOG		10000000	//10M, k125so.
#define CNT_KEEPING_TRS		100 

//qapi.h
#define CNT_RETRY_NET_PING		2
#define TM_INTER_CLI_DEAD_CHECK	1
#define MSEC_TMOUT_PING			100 //k14so. 1000

//qsrv.h
#define TM_CHECK_DEAD_SESSION	1
#define TMOUT_DEAD_SESSION		30	//2
#define TMWAIT_FORCE_CLOSE	2

#define CNT_CONNECT_GABAGE	5

#define CNT_TMOUT_CHECK_RESET	4

#define CNT_FREE_WORK_VECT_MEM 10000

#define TM_WAIT_OVERLAP_SOCDESC	1

#define RECYCLE_CAPACITY	1000000  //1M

#define URL_NAME_LEN	4096

#define NOT_INDEX_COUNT		10000

#define TM_WAIT_DDL_REROAD	3 //이 시간이 짧아 저장중에 리로드되면 크래쉬될수있다.

#define TM_DELAY_REPLICATE_TRANS 5

#define RECONNECT_FOWARD_GATEWAY	-2

#define TRSMDA_RECYCLE_MEM			0x200000 //0x4000000 //64M, k369so.

#define NUM_PROSEC	32 //64 //16 //4 //32	// real process number(기본적으로는 실 코어 갯수로 설정), 기본 구간을 생성할때 
//이 갯수만큼 기본구간을 생성한다. SEM_MAX_COUNT보다는 작아야하고 2의 승수로 정해야 한다. 이유는 기존 구간있어 재사용하거나 
//테이블 페이지와 같이 한번 구간 분리되면 바뀌지 않는 경우 ReserveProsecution로 proc을 예약할때 요청 갯수보다 작을 경우에도
//2의 승수로 하여 기존 구간을 몇개씩 실행할때 기존 구간을 그대로 사용할수있기때문.-> CALC_NUM_PART_PROC에서 첫번째 그룹에 나머지 구간들을 더 추가하므로 2승수 필요없음.

//sdb는 싱크에 사용되는 디비이므로 이 디비의 페이지 사이즈는 여기에 둔다.
#define MGRP_SDB_SIZE		PROVIS_C_CLASS
#define MGRP_MDB_SIZE		MGRP_SDB_SIZE
#define CINDEX_BLOCK_SIZE	0xFFD0	//k395so.C_SEG_BLOCK - IndexHead size

#define IPS_CPACK_MARK			1
#define REGISTER_CPACK_MARK		2
#define TABLE_CPACK_MARK		3
#define SMAT_CPACK_MARK			4
#define JSTACK_CPACK_MARK		5
#define JLAP_CPACK_MARK			6
#define JMAT_CPACK_MARK			7
#define OPS_CPACK_MARK			8
#define SUPREL_CPACK_MARK		9

#define BEG_SMAT_COL		(ReferColumn *)1

#define CROSS_JOIN_IDX_WIDTH	1
#define REF_COL_WIDTH		sizeof(intx)
#define SUB_SMAT_COL_WIDTH	sizeof(intx)
#define IP_WIDTH			sizeof(intx)
#define OPR_STATCK_WIDTH	16
#define VECT_HEAD_WIDTH		sizeof(ByVector)

#define N_COLLECT_FREE_PAGE	5 //1 //k375so_2.테스트용 갯수, 나중에 늘려야

#define N_MAX_COLUMN	580
#define N_MAX_FROM_TAB	100

#define NUM_OUT_ROWS	300
#define NUM_JOIN_ROWS	20000
#define CNT_REENT_HEART_BIT		100
#define RECYCLE_REENT_OR_OUTJ 100000

#define IGNORE_DIFF_SIZE	30

#define NUM_PAGE_PART_PROPA	1000 //60 M
#define NUM_ROW_PART_PROPA	1000000
