#ifndef _H_STACK
#define _H_STACK
//스택의 탑을 스캔하는 속도를 위해서 푸쉬할때 스택포인터를 먼저 증가한후 데이터를 푸쉬한다.
//따라서 푸쉬단위를 싱글사이즈 아니면 더블사이즈로 정하여 하나의 사이즈로만 푸쉬해야한다.
#include "misc/xt.h"

#define StackPointer(pstack) pstack->headStack
#define RetroStack(pstack, sp) pstack->headStack = sp
#define IsNullStack(pstack) pstack->headStack == 0

#define DecreaseStack(pstack, cnt) pstack->headStack -= cnt
#define IncreaseStack(pstack, cnt) pstack->headStack += cnt
//스텍포인터를 cnt만큼 감소하여 언더플로우 체크
#define DecChkStack(pstack, cnt) {\
	DecreaseStack(pstack, cnt);\
	if(pstack->headStack < 0) throwExpt(-1, "DecChkStack underflow\n");\
}
//스텍포인터를 cnt만큼 증가하여 오버플로우 체크
#define IncChkStack(pstack, cnt) {\
	IncreaseStack(pstack, cnt);\
	if(pstack->headStack == pstack->szStack) throwExpt(-1, "IncChkStack overflow\n");\
}
/******************* 이하는 스택 스캔 오퍼레인터(스택 포인터 증감이 아니라) *****************/
#define UnderChkStack(pstack) DecChkStack(pstack, 1) 
#define UnderDChkStack(pstack) DecChkStack(pstack, 2)
//스택의 바닥으로부터 인덱스 옵셋 만큼의 스택 프레임 포인터(스택프레임 자체를 데이터로 사용할때)
#define BottomItStack_(pstack, idx) (pstack->stackFrame + idx)
//스택의 바닥으로부터 인덱스 옵셋 만큼의 스택 값
#define BottomItStack(pstack, idx) *BottomItStack_(pstack, idx)
//스택의 탑으로부터 dec만큼 밑의 스택 프레임 포인터
#define TopItStack_(pstack, dec) BottomItStack_(pstack, (pstack->headStack - dec))
//스택의 탑으로부터 dec만큼 밑의 스택 값
#define TopItStack(pstack, dec) *TopItStack_(pstack, dec)
//unit 포인터로 캐스트하는 부분에서 경고에러가 나지만 타입을 맞추기위해 어쩔수 없음
#define HeadItStack_(pstack) BottomItStack_(pstack, pstack->headStack)
#define HeadItStack(pstack) *HeadItStack_(pstack)
#define PointHeadStack(pstack) (bytex *)HeadItStack_(pstack)
//현 스택포인터로부터 인덱스만큼 위의 스택 프레임 포인터
#define CurrentItStack_(pstack, i) BottomItStack_(pstack, (pstack->headStack + i))
//현 스택포인터로부터 인덱스만큼 위의 스택 값
#define CurrentItStack(pstack, i) BottomItStack(pstack, (pstack->headStack + i))
//스택의 탑으로부터 dec만큼 밑의 스택 프레임 포인터를 언더플로우 체크를 하면서 획득
#define TopItChkStack_(pstack, dec, pv) {\
	if((pstack->headStack - dec) < 0) throwExpt(-1, "TopItChkStack_ underflow\n");\
	pv = TopItStack_(pstack, dec);\
}
#define TopItScanStack_(tp, pstack, dec, pv) {\
	if((pstack->headStack - dec) == 0) pv = nullx;\
	else pv = (tp *)TopItStack_(pstack, dec);\
}
//스택의 탑으로부터 dec만큼 밑의 스택 값을 언더플로우 체크를 하면서 획득(스캔)
#define TopItChkStack(pstack, dec, v) {\
	if((pstack->headStack - dec) < 0) throwExpt(-1, "TopItChkStack_ underflow\n");\
	v = TopItStack(pstack, dec);\
}
/***************************** 이하는 스택 포인터 증감 오퍼레이터 **********************/
//현 스택 포인터의 스택 값을 획득하고 스택 포인터를 하나 감소시킨다.
#define PopStack(pstack, v) {\
	v = HeadItStack(pstack);\
	DecChkStack(pstack, 1);\
}
//스택포인터를 먼저 증가하고 값을 넣는다.
#define PushPStack(pstack, pv) {\
	IncChkStack(pstack, 1);\
	pv = HeadItStack_(pstack);\
}
//현 스택 포인터의 스택 프레임의 포인터를 획득하고 스택 포인터를 하나 감소시킨다.
#define PopPStack(pstack, pv) {\
	pv = HeadItStack_(pstack);\
	DecChkStack(pstack, 1);\
}

#define PopTStack(tp, pstack, pv) {\
	pv = (tp *)HeadItStack(pstack);\
	DecChkStack(pstack, 1);\
}
//스택값이 있는지를 먼저 체크하고 있으면 값을 리턴한다.
#define PopTStack2(tp, pstack, v) {\
	if(IsNullStack(pstack)) v = 0;\
	else v = (tp *)HeadItStack(pstack);\
}
//스택포인터를 먼저 증가하고 값을 넣는다.
#define PushStack(pstack, v) {\
	IncChkStack(pstack, 1);\
	HeadItStack(pstack) = v;\
}

#define PushNChkStack(pstack, v) {\
	IncreaseStack(pstack, 1);\
	HeadItStack(pstack) = v;\
}
#define PopNChkStack(pstack, v) {\
	v = HeadItStack(pstack);\
	DecreaseStack(pstack, 1);\
}

//스택의 탑을 복사하여 하나더 위에 쌓는다.
#define LiftStack(pstack) {\
	IncChkStack(pstack, 1);\
	HeadItStack(pstack) = TopItStack(pstack, 1);\
}
//스택의 탑을 뒤집는다.
#define ReverseStack(pstack, tmp) {\
	tmp = HeadItStack(pstack);\
	HeadItStack(pstack) = TopItStack(pstack, 1);\
	TopItStack(pstack, 1) = tmp;\
}

//이하는 더블사이즈(16바이트) 오퍼레이션, 스택 프레임의 포인터를 반환한다는 것에 주의

//스택포인터를 먼저 증가하고 값을 넣는다.
#define PushDoubleStack(tp, pstack, v) {\
	IncChkStack(pstack, 2);\
	*(tp *)HeadItStack_(pstack) = v;\
}
#define PopDoubleStack(tp, pstack, v) {\
	v = *(tp*)HeadItStack_(pstack);\
	DecChkStack(pstack, 2);\
}
#define PushPointStack(tp, pstack, pv) {\
	IncChkStack(pstack, 2);\
	pv = (tp *)HeadItStack_(pstack);\
}
#define PopPointStack(tp, pstack, pv) {\
	pv = (tp *)HeadItStack_(pstack);\
	DecChkStack(pstack, 2);\
}
//스택의 탑을 복사하여 하나더 위에 쌓는다.
#define LiftDoubleStack(tp, pstack) {\
	IncChkStack(pstack, 2);\
	*(tp *)HeadItStack_(pstack) = *(tp *)TopItStack_(pstack, 2);\
}
//스택의 탑을 더블사이즈 증가한후 탑으로 부터 n_dec(더블단위-16바이트)번째 밑 스택에서 더블단위 하나위 스택으로 n_dec 더블스택 슬롯 사이즈 만큼 이동시킨후
//n_dec번째 스롯 포인터를 리턴한다. 즉, n_dec(더블단위, 1이면 16바이트)만큼 스택 탑으로 부터의 밑의 스택슬롯을 더블 사이즈만큼 벌린다.
#define GapDoubleStack(pstack, n_dec, pv) {\
	IncChkStack(pstack, 2);\
	pv = TopItStack_(pstack, n_dec * 2);\
	memmove(pv +2, pv, 16 * n_dec);\
}
//스택의 탑을 뒤집는다.
#define ReverseDoubleStack(tp, pstack, tmp) {\
	tmp = *(tp *)HeadItStack_(pstack);\
	*(tp *)HeadItStack_(pstack) = *(tp *)TopItStack_(pstack, 2);\
	*(tp *)TopItStack_(pstack, 2) = tmp;\
}
//스택의 탑을 두개 뒤집는다.
#define ReverseDoubleStack2(tp, pstack, tmp) {\
	tmp = *(tp *)HeadItStack_(pstack);\
	*(tp *)HeadItStack_(pstack) = *(tp *)TopItStack_(pstack, 2);\
	*(tp *)TopItStack_(pstack, 2) = *(tp *)TopItStack_(pstack, 4);\
	*(tp *)TopItStack_(pstack, 4) = tmp;\
}
#define IsEmptyStack(pstack) (pstack->headStack == 0)

#define _InitStack(pstack) {\
	pstack->headStack = 0;\
	BottomItStack(pstack, 0) = 0;\
}
#define InitStack(pstack, pframe, sz) {\
	pstack->szStack = sz;\
	pstack->stackFrame = pframe;\
	_InitStack(pstack);\
	BottomItStack(pstack, 1) = 0;\
}
#define InitPStack(pstack, pframe, sz) {\
	pstack->szStack = sz;\
	pstack->stackFrame = pframe;\
	pstack->headStack = 0;\
}
#define ResetStack(pstack) pstack->headStack = 0
//밸류 스택은 아래와같이 각 원시타입별로 스택 자료구조를 정의하고 위 매크로를 바로 사용한다.
typedef struct {
	intx szStack;
	intx headStack;
	intx *stackFrame;
} IntStack;
typedef struct {
	intx szStack;
	intx headStack;
	floatx *stackFrame;
} FloatStack;
typedef struct {
	intx szStack;
	intx headStack;
	unit *stackFrame;
} UnitStack;
//리스트로 운용되는 스택, 스택해드 초기화 및 푸쉬함수를 래핑하여 스택노드 할당하는 것을 작성해야함
typedef struct StackNode_ {
	void *stackTag;
	struct StackNode_ *ptrLeft, *ptrRight;\
} StackNode;
#define PushLstStack_(head, val) {\
	head->ptrLeft = head->ptrLeft->ptrRight;\
	head->ptrLeft->stackTag = val;\
}
#define PopLstStack(tp, head, val) {\
	if(head->ptrLeft == head) val = nullx;\
	else {\
		val = (tp *)head->ptrLeft->stackTag;\
		head->ptrLeft = head->ptrLeft->ptrLeft;\
	}\
}
		
#endif
